import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import nest_asyncio
import numpy as np
import torch
from stable_baselines3.dqn import DQN as SB3DQN
from stable_baselines3.dqn.policies import MultiInputPolicy as DQNMultiInputPolicy

MASK_NEG_LARGE = -1e9

class MaskedDQNPolicy(DQNMultiInputPolicy):
    """DQN policy with action masking on Q-values before argmax."""

    def _predict(self, observation, deterministic=True):
        q_values = self.q_net(observation)
        action_mask = observation["action_mask"]
        masked_q = q_values + (action_mask - 1.0) * abs(MASK_NEG_LARGE)
        action = masked_q.argmax(dim=1).reshape(-1)
        return action


class MaskedDQN(SB3DQN):
    """DQN with masked exploration sampling compatible with VecEnv."""

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # En exploracion, muestrea SOLO acciones validas y devuelve arreglo por entorno.
        if (not deterministic) and np.random.rand() < self.exploration_rate:
            if isinstance(observation, dict) and "action_mask" in observation:
                mask = np.asarray(observation["action_mask"])
                if mask.ndim == 1:
                    mask = mask[None, :]

                actions = []
                for i in range(mask.shape[0]):
                    valid = np.flatnonzero(mask[i] > 0.5)
                    if len(valid) > 0:
                        actions.append(int(np.random.choice(valid)))
                    else:
                        actions.append(int(self.action_space.sample()))

                return np.asarray(actions), state

            # Fallback cuando no llega mascara (no deberia pasar con wrapper)
            n_envs = 1
            if isinstance(observation, dict):
                first_key = next(iter(observation.keys()))
                first_value = np.asarray(observation[first_key])
                if first_value.ndim >= 1:
                    n_envs = int(first_value.shape[0])

            random_actions = np.asarray([int(self.action_space.sample()) for _ in range(n_envs)])
            return random_actions, state

        # En explotacion usa la logica normal (que ya pasa por la policy enmascarada)
        return super().predict(observation, state, episode_start, deterministic)
    def _sample_action(self, learning_starts, action_noise=None, n_envs=None):
        # Exploration path: sample only valid actions from action_mask.
        if self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate:
            obs = self._last_obs
            if isinstance(obs, dict) and "action_mask" in obs:
                mask = np.asarray(obs["action_mask"])
                if mask.ndim == 1:
                    mask = mask[None, :]

                actions = []
                for i in range(mask.shape[0]):
                    valid = np.flatnonzero(mask[i] > 0.5)
                    if len(valid) > 0:
                        actions.append(int(np.random.choice(valid)))
                    else:
                        # Fallback defensive: choose fast action when mask is malformed.
                        actions.append(0)

                actions = np.asarray(actions)
            else:
                env_count = n_envs if n_envs is not None else self.n_envs
                actions = np.asarray([int(self.action_space.sample()) for _ in range(env_count)])

            return actions, actions

        # Exploitation path uses masked policy.
        actions, _ = self.predict(self._last_obs, deterministic=True)
        return actions, actions

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Same as SB3 DQN train, but mask next-state Q-values before max().
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                next_q_values_all = self.q_net_target(replay_data.next_observations)

                if isinstance(replay_data.next_observations, dict) and "action_mask" in replay_data.next_observations:
                    next_mask = replay_data.next_observations["action_mask"]
                    valid_any = (next_mask > 0.5).any(dim=1, keepdim=True)

                    masked_next_q = next_q_values_all.masked_fill(next_mask < 0.5, -1e9)
                    masked_next_q_max = masked_next_q.max(dim=1, keepdim=True).values
                    unmasked_next_q_max = next_q_values_all.max(dim=1, keepdim=True).values

                    # If a row has an invalid all-zero mask, avoid injecting huge negative targets.
                    next_q_values = torch.where(valid_any, masked_next_q_max, unmasked_next_q_max)
                else:
                    next_q_values = next_q_values_all.max(dim=1, keepdim=True).values

                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))




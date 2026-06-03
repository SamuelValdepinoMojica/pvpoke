import csv
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from environment.ClassPVPOKE import PVPokeEnv
except ModuleNotFoundError:
    from ClassPVPOKE import PVPokeEnv

logger = logging.getLogger(__name__)


@dataclass
class QLearningConfig:
    ws_uri: str = "ws://localhost:8000/ws"
    client_id: str = "rl_0"
    target_id: str = "pvpoke_0"
    battle_format: str = "1v1"
    observation_mode: str = "minimal"
    preprocess_mode: str = "discrete"
    discretization_bins: int = 10
    reset_random: bool = False
    use_mask: bool = True
    n_training_episodes: int = 20_000
    eval_interval: int = 100
    n_eval_episodes: int = 30
    max_steps: int = 500
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.99
    run_prefix: str = "ql"
    run_name: Optional[str] = None
    output_dir: Path = Path("outputs/qlearning")
    seed: Optional[int] = None
    show_progress: bool = True


class MaskedQLearner:
    """Tabular Q-Learning that respects action masks when use_mask=True."""

    def __init__(
        self,
        env: PVPokeEnv,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.state_bins = self._compute_effective_state_bins()
        self.state_shape = tuple(int(v) for v in self.state_bins.tolist())

        self.state_space_size = 1
        for bins in self.state_shape:
            self.state_space_size *= bins

        self.action_space_size = int(env.action_space.n)

        max_dense_states = 5_000_000
        if self.state_space_size > max_dense_states:
            raise ValueError(
                f"State space too large ({self.state_space_size:,}). "
                "Use observation_mode='minimal' and preprocess_mode='discrete'."
            )

        self.Q = np.zeros((self.state_space_size, self.action_space_size), dtype=np.float32)
        self.visit_counts = np.zeros(
            (self.state_space_size, self.action_space_size),
            dtype=np.int32,
        )
        self.invalid_action_attempts = 0

        logger.info(
            "QLearner ready | states=%s | actions=%s | alpha=%.3f | gamma=%.3f | eps=%.3f",
            f"{self.state_space_size:,}",
            self.action_space_size,
            self.alpha,
            self.gamma,
            self.epsilon,
        )

    def _compute_effective_state_bins(self) -> np.ndarray:
        bins = np.asarray(self.env.observation_space.high + 1, dtype=np.int64)
        bins = np.maximum(bins, 1)

        if getattr(self.env, "preprocess_mode", None) != "discrete":
            return bins

        obs_mode = getattr(self.env, "observation_mode", "full")
        num_pokemon = int(getattr(self.env, "num_pokemon", 1))

        if obs_mode == "minimal":
            ally_shield_idx = 2 * num_pokemon
            enemy_shield_idx = (2 * num_pokemon + 1) + 2 * num_pokemon

            if ally_shield_idx < bins.shape[0]:
                bins[ally_shield_idx] = 3
            if enemy_shield_idx < bins.shape[0]:
                bins[enemy_shield_idx] = 3

            return bins

        ally_attrs = int(getattr(self.env, "ally_attrs_per_pokemon", 0))
        enemy_attrs = int(getattr(self.env, "enemy_attrs_per_pokemon", 0))
        battle_format = getattr(self.env, "battle_format", "1v1")

        idx = 0
        idx += num_pokemon * ally_attrs
        if idx < bins.shape[0]:
            bins[idx] = 3
        idx += 1

        if battle_format == "3v3":
            idx += num_pokemon
            if idx < bins.shape[0]:
                bins[idx] = num_pokemon + 1
            idx += 1

        idx += num_pokemon * enemy_attrs
        if idx < bins.shape[0]:
            bins[idx] = 3
        idx += 1

        if battle_format == "3v3":
            idx += num_pokemon
            if idx < bins.shape[0]:
                bins[idx] = num_pokemon + 1
            idx += 1

        return bins

    def discretize_state(self, state: np.ndarray) -> tuple:
        state_arr = np.asarray(state, dtype=np.int64)
        state_arr = np.clip(state_arr, 0, self.state_bins - 1)
        return tuple(state_arr.tolist())

    def choose_action_masked(self, state: tuple, mask: np.ndarray, training: bool = True) -> int:
        state_index = np.ravel_multi_index(state, self.state_shape)
        mask_arr = np.asarray(mask).reshape(-1)
        valid_actions = np.where(mask_arr > 0.5)[0]

        if len(valid_actions) == 0:
            logger.warning("All-zero action mask encountered; using action 0 as fallback.")
            return 0

        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        q_values_valid = self.Q[state_index, valid_actions]
        best_valid_idx = int(np.argmax(q_values_valid))
        return int(valid_actions[best_valid_idx])

    def update_q_table(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_mask: np.ndarray,
        done: bool,
    ) -> float:
        state_index = np.ravel_multi_index(state, self.state_shape)
        next_state_index = np.ravel_multi_index(next_state, self.state_shape)

        valid_next_actions = np.where(np.asarray(next_mask).reshape(-1) > 0.5)[0]
        if len(valid_next_actions) > 0:
            max_next_q = float(np.max(self.Q[next_state_index, valid_next_actions]))
        else:
            max_next_q = 0.0

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * max_next_q

        action = int(action)
        td_error = td_target - float(self.Q[state_index, action])
        self.Q[state_index, action] += self.alpha * td_error
        self.visit_counts[state_index, action] += 1

        return abs(float(td_error))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_coverage(self) -> float:
        visited = int(np.count_nonzero(self.visit_counts))
        total = int(self.state_space_size * self.action_space_size)
        return (visited / total) * 100.0 if total else 0.0


def run_q_episode(
    learner: MaskedQLearner,
    training: bool = True,
    use_mask: bool = True,
    max_steps: int = 500,
) -> Dict[str, Any]:
    state, _ = learner.env.reset()
    state = learner.discretize_state(state)
    mask = np.asarray(learner.env.action_masks(), dtype=np.float32)

    episode_reward = 0.0
    episode_td_errors = []
    invalid_selected = 0
    steps = 0
    done = False

    while (not done) and (steps < max_steps):
        state_index = np.ravel_multi_index(state, learner.state_shape)

        if use_mask:
            action = learner.choose_action_masked(state, mask, training=training)
        else:
            if training and np.random.rand() < learner.epsilon:
                action = int(np.random.randint(learner.action_space_size))
            else:
                action = int(np.argmax(learner.Q[state_index]))

            if mask[action] <= 0.5:
                invalid_selected += 1

        next_state, reward, terminated, truncated, _ = learner.env.step(action)
        next_state = learner.discretize_state(next_state)
        next_mask = np.asarray(learner.env.action_masks(), dtype=np.float32)
        done = bool(terminated or truncated)

        if training:
            if use_mask:
                td_error = learner.update_q_table(
                    state,
                    action,
                    float(reward),
                    next_state,
                    next_mask,
                    done,
                )
            else:
                next_state_index = np.ravel_multi_index(next_state, learner.state_shape)
                if done:
                    td_target = float(reward)
                else:
                    max_next_q = float(np.max(learner.Q[next_state_index]))
                    td_target = float(reward) + learner.gamma * max_next_q

                td_error = td_target - float(learner.Q[state_index, action])
                learner.Q[state_index, action] += learner.alpha * td_error
                learner.visit_counts[state_index, action] += 1
                td_error = abs(float(td_error))

            episode_td_errors.append(float(td_error))

        episode_reward += float(reward)
        state = next_state
        mask = next_mask
        steps += 1

    if training:
        learner.decay_epsilon()

    return {
        "reward": float(episode_reward),
        "steps": int(steps),
        "avg_td_error": float(np.mean(episode_td_errors)) if episode_td_errors else 0.0,
        "invalid_selected": int(invalid_selected),
        "ended_by_step_limit": bool((not done) and (steps >= max_steps)),
    }


def evaluate_policy(
    learner: MaskedQLearner,
    use_mask: bool = True,
    n_episodes: int = 5,
    max_steps: int = 500,
) -> Dict[str, Any]:
    old_epsilon = float(learner.epsilon)
    learner.epsilon = 0.0

    rewards = []
    wins = 0
    steps = []
    invalid = []

    for _ in range(n_episodes):
        stats = run_q_episode(
            learner,
            training=False,
            use_mask=use_mask,
            max_steps=max_steps,
        )
        rewards.append(stats["reward"])
        steps.append(stats["steps"])
        invalid.append(stats["invalid_selected"])
        if stats["reward"] > 0:
            wins += 1

    learner.epsilon = old_epsilon

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "win_rate": float(wins / n_episodes) if n_episodes > 0 else 0.0,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "mean_invalid_selected": float(np.mean(invalid)) if invalid else 0.0,
        "rewards": rewards,
    }


def _build_env(config: QLearningConfig) -> PVPokeEnv:
    env = PVPokeEnv(
        config.ws_uri,
        config.client_id,
        config.target_id,
        battle_format=config.battle_format,
        reset_random=config.reset_random,
        observation_mode=config.observation_mode,
        preprocess_mode=config.preprocess_mode,
        discretization_bins=config.discretization_bins,
    )
    env.loop.run_until_complete(env.connect())
    return env


def _write_summary_csv(summary: Dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fieldnames = list(summary.keys())

    if not write_header:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
        if existing:
            fieldnames = existing

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)


def run_qlearning_experiment(config: QLearningConfig) -> Dict[str, Any]:
    if config.preprocess_mode != "discrete":
        logger.warning("Q-learning is typically used with preprocess_mode='discrete'.")

    logger.info(
        "Config | format=%s | obs=%s | preprocess=%s | bins=%s | episodes=%s | eval_interval=%s | max_steps=%s | mask=%s",
        config.battle_format,
        config.observation_mode,
        config.preprocess_mode,
        config.discretization_bins,
        config.n_training_episodes,
        config.eval_interval,
        config.max_steps,
        config.use_mask,
    )

    if config.seed is not None:
        np.random.seed(int(config.seed))

    mode = "mask" if config.use_mask else "no_mask"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"{config.run_prefix}_bins{config.discretization_bins}_{mode}_{timestamp}"

    env = None
    learner = None

    try:
        env = _build_env(config)
        logger.info(
            "Env | obs_space=%s | obs_shape=%s | action_n=%s",
            env.observation_space,
            getattr(env.observation_space, "shape", None),
            int(env.action_space.n),
        )
        learner = MaskedQLearner(
            env,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_min=config.epsilon_min,
            epsilon_decay=config.epsilon_decay,
        )
        state_space_size = int(learner.state_space_size)

        train_rewards = []
        train_td_errors = []
        train_steps = []
        train_invalid = []
        eval_history = []

        start_t = time.time()
        progress = None
        if config.show_progress:
            if tqdm is None:
                logger.info("Progress bar disabled (tqdm not installed).")
            else:
                progress = tqdm(
                    total=config.n_training_episodes,
                    desc="QLearning",
                    unit="ep",
                    ncols=100,
                )

        for episode in range(1, config.n_training_episodes + 1):
            stats = run_q_episode(
                learner,
                training=True,
                use_mask=config.use_mask,
                max_steps=config.max_steps,
            )
            train_rewards.append(stats["reward"])
            train_td_errors.append(stats["avg_td_error"])
            train_steps.append(stats["steps"])
            train_invalid.append(stats["invalid_selected"])

            if (episode % config.eval_interval == 0) or (episode == config.n_training_episodes):
                eval_stats = evaluate_policy(
                    learner,
                    use_mask=config.use_mask,
                    n_episodes=config.n_eval_episodes,
                    max_steps=config.max_steps,
                )
                eval_stats["episode"] = int(episode)
                eval_history.append(eval_stats)

                logger.info(
                    "Ep %s/%s | TrainR=%.2f | TD=%.5f | Inv=%.2f | EvalWin=%.1f%%",
                    episode,
                    config.n_training_episodes,
                    float(np.mean(train_rewards[-config.eval_interval :])),
                    float(np.mean(train_td_errors[-config.eval_interval :])),
                    float(np.mean(train_invalid[-config.eval_interval :])),
                    float(eval_stats["win_rate"] * 100.0),
                )

                if progress is not None:
                    progress.set_postfix(
                        {
                            "trainR": f"{float(np.mean(train_rewards[-config.eval_interval:])):.2f}",
                            "evalWin%": f"{float(eval_stats['win_rate'] * 100.0):.1f}",
                            "inv": f"{float(np.mean(train_invalid[-config.eval_interval:])):.2f}",
                        }
                    )

            if progress is not None:
                progress.update(1)

        if progress is not None:
            progress.close()

        duration = time.time() - start_t
        final_eval = eval_history[-1] if eval_history else {}
        coverage = learner.get_coverage()

        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        qtable_path = out / f"{run_name}_qtable.npz"
        np.savez_compressed(
            qtable_path,
            Q=learner.Q,
            state_bins=learner.state_bins,
            state_shape=np.array(learner.state_shape, dtype=np.int64),
            action_space_size=np.array([learner.action_space_size], dtype=np.int64),
        )

        history_path = out / f"{run_name}_history.json"
        history_payload = {
            "run_name": run_name,
            "config": {
                "bins": int(config.discretization_bins),
                "use_mask": bool(config.use_mask),
                "n_training_episodes": int(config.n_training_episodes),
                "eval_interval": int(config.eval_interval),
                "n_eval_episodes": int(config.n_eval_episodes),
                "max_steps": int(config.max_steps),
                "alpha": float(config.alpha),
                "gamma": float(config.gamma),
                "epsilon_start": float(config.epsilon),
                "epsilon_min": float(config.epsilon_min),
                "epsilon_decay": float(config.epsilon_decay),
                "reset_random": bool(config.reset_random),
                "client_id": config.client_id,
                "target_id": config.target_id,
                "battle_format": config.battle_format,
                "observation_mode": config.observation_mode,
                "preprocess_mode": config.preprocess_mode,
            },
            "train": {
                "rewards": [float(x) for x in train_rewards],
                "td_errors": [float(x) for x in train_td_errors],
                "steps": [int(x) for x in train_steps],
                "invalid_selected": [int(x) for x in train_invalid],
            },
            "eval": eval_history,
            "summary": {
                "duration_sec": float(duration),
                "coverage": float(coverage),
                "final_eval_mean_reward": float(final_eval.get("mean_reward", 0.0)),
                "final_eval_win_rate": float(final_eval.get("win_rate", 0.0)),
                "final_eval_mean_steps": float(final_eval.get("mean_steps", 0.0)),
                "avg_train_reward_last_5": float(np.mean(train_rewards[-5:])) if train_rewards else 0.0,
                "avg_train_invalid_selected": float(np.mean(train_invalid)) if train_invalid else 0.0,
            },
        }
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history_payload, f, ensure_ascii=False, indent=2)

        summary = {
            "run_name": run_name,
            "bins": int(config.discretization_bins),
            "use_mask": bool(config.use_mask),
            "state_space": int(state_space_size),
            "qtable_path": str(qtable_path),
            "history_path": str(history_path),
            "coverage": float(history_payload["summary"]["coverage"]),
            "final_eval_mean_reward": float(history_payload["summary"]["final_eval_mean_reward"]),
            "final_eval_win_rate": float(history_payload["summary"]["final_eval_win_rate"]),
            "avg_train_reward_last_5": float(history_payload["summary"]["avg_train_reward_last_5"]),
            "avg_train_invalid_selected": float(history_payload["summary"]["avg_train_invalid_selected"]),
            "duration_sec": float(history_payload["summary"]["duration_sec"]),
        }

        _write_summary_csv(summary, out / "summary.csv")

        summary_path = out / f"{run_name}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Saved qtable: %s", qtable_path)
        logger.info("Saved history: %s", history_path)
        logger.info("Saved summary: %s", summary_path)

        return summary

    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_err:
                logger.warning("Error closing env: %s", close_err)
        if learner is not None:
            del learner
        if env is not None:
            del env
        gc.collect()

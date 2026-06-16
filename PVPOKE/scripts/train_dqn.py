import argparse
import logging
import sys
import os
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Añadir el directorio raíz al path para que funcionen los imports absolutos
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Importamos las herramientas de Stable Baselines 3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Importamos tu entorno y tu agente personalizado
from environment.ClassPVPOKE import PVPokeEnv
from rl.masked_dqn import MaskedDQN, MaskedDQNPolicy

# ==========================================
# Action Masking Environment Wrapper
# ==========================================
class MaskedDictObservationWrapper(gym.Wrapper):
    """
    Transforms the observation space into a Dict containing both 
    the raw game state and the active action mask.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "obs": env.observation_space,
            "action_mask": spaces.Box(low=0.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32),
        })

    def _obs_with_mask(self, obs):
        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "action_mask": np.asarray(self.env.action_masks(), dtype=np.float32),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs_with_mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs_with_mask(obs), reward, terminated, truncated, info

# ==========================================
# Main Orchestration & Hyperparameters
# ==========================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stable Baselines 3 Masked DQN for PvPoke.")
    
    # Connection Arguments
    parser.add_argument("--ws-uri", default="ws://localhost:8000/ws", help="WebSocket URI for the PvPoke server")
    parser.add_argument("--client-id", default=None, help="Custom Client ID")
    parser.add_argument("--target-id", default=None, help="Custom Target ID")
    parser.add_argument("--pair-id", type=int, default=0, help="ID to generate default client/target pairs")
    
    # Environment Arguments
    parser.add_argument("--battle-format", default="1v1", choices=["1v1", "3v3"], help="Battle format to simulate")
    parser.add_argument("--observation-mode", default="minimal", choices=["minimal", "full"], help="State representation mode")
    
    # DQN Hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Size of the replay buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size for each gradient update")
    parser.add_argument("--exploration-fraction", type=float, default=0.15, help="Fraction of entire training period over which the exploration rate is reduced")
    
    # Saving & Evaluation Config
    parser.add_argument("--save-freq", type=int, default=20_000, help="Save a checkpoint every X steps")
    parser.add_argument("--eval-freq", type=int, default=0, help="Evaluate every X steps. Set to 0 to disable or use --n-evals instead.")
    parser.add_argument("--n-evals", type=int, default=5, help="Total number of evaluations during training. Overrides --eval-freq if > 0.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes to play during each evaluation.")
    parser.add_argument("--run-name", default="dqn_run", help="Name of the current experiment run")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs" / "dqn"), help="Base directory for saving models and logs")
    
    return parser

def resolve_ids(pair_id: int, client_id: str, target_id: str) -> tuple[str, str]:
    final_client = client_id if client_id else f"rl_{pair_id}"
    final_target = target_id if target_id else f"pvpoke_{pair_id}"
    return final_client, final_target

def make_env(ws_uri, client_id, target_id, format, obs_mode):
    """Creates the base environment and applies the required Wrappers."""
    base_env = PVPokeEnv(
        ws_uri, 
        client_id, 
        target_id, 
        battle_format=format, 
        observation_mode=obs_mode,
        reset_random=False
    )
    base_env.loop.run_until_complete(base_env.connect())
    return Monitor(MaskedDictObservationWrapper(base_env))

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = build_arg_parser().parse_args()

    # Directory setup
    model_dir = os.path.join(args.output_dir, args.run_name, "models")
    tb_dir = os.path.join(args.output_dir, args.run_name, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    client_id, target_id = resolve_ids(args.pair_id, args.client_id, args.target_id)

    logging.info(f"Starting DQN Training ({args.total_timesteps} steps)")
    logging.info(f"Connecting to {args.ws_uri} | client={client_id}, target={target_id}")

    # 1. Initialize ONE single Environment instance
    env = make_env(args.ws_uri, client_id, target_id, args.battle_format, args.observation_mode)

    try:
        # 2. Build the Custom Masked DQN Model
        model = MaskedDQN(
            policy=MaskedDQNPolicy,
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=10_000,
            batch_size=args.batch_size,
            gamma=0.999,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=3000,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            tensorboard_log=tb_dir,
            policy_kwargs={"net_arch": [128, 64, 32]},
            verbose=1,
        )

        # 3. Setup Callbacks
        callbacks = []
        
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=model_dir,
            name_prefix="dqn_ckpt",
        )
        callbacks.append(checkpoint_callback)

        # Calculate Evaluation Frequency
        eval_freq = args.eval_freq
        if args.n_evals > 0:
            eval_freq = args.total_timesteps // args.n_evals
            logging.info(f"Auto-calculated evaluation frequency: Every {eval_freq} steps (Total: {args.n_evals} evaluations).")

        # Inject Intermediate Evaluation if frequency > 0
        if eval_freq > 0:
            eval_callback = EvalCallback(
                env, # Utiliza el MISMO entorno para evaluar
                best_model_save_path=model_dir,
                log_path=model_dir,
                eval_freq=eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
            logging.info(f"Intermediate evaluations ENABLED: Playing {args.eval_episodes} episodes every {eval_freq} steps.")
        else:
            logging.info("Intermediate evaluations DISABLED.")

        # 4. Execute Training Loop
        start_time = time.time()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        duration = time.time() - start_time

        # 5. Save Final Model
        final_path = os.path.join(model_dir, "dqn_final_model")
        model.save(final_path)
        logging.info(f"Training completed in {duration/60:.2f} minutes. Model saved to {final_path}")

        # 6. Post-Training Evaluation
        logging.info("Starting final post-training evaluation (30 episodes)...")
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=30, 
            deterministic=True
        )
        logging.info(f"Final Evaluation Results: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    finally:
        # Safely close the WebSocket connection
        env.close()
        logging.info("Environment connection closed safely.")

if __name__ == "__main__":
    main()
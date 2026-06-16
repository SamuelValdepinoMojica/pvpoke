import argparse
import logging
import sys
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Add root directory to sys.path for absolute imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Stable Baselines 3 imports
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# sb3_contrib specific imports for Maskable PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

# Custom Environment import
from environment.ClassPVPOKE import PVPokeEnv

# ==========================================
# Action Masking Helper Function
# ==========================================
def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Function required by sb3_contrib's ActionMasker.
    It retrieves the boolean/binary array of valid actions from the base environment.
    """
    return env.action_masks()


# ==========================================
# Main Orchestration & Hyperparameters
# ==========================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sb3_contrib MaskablePPO for PvPoke.")
    
    # Connection Arguments
    parser.add_argument("--ws-uri", default="ws://localhost:8000/ws", help="WebSocket URI for the PvPoke server")
    parser.add_argument("--client-id", default=None, help="Custom Client ID")
    parser.add_argument("--target-id", default=None, help="Custom Target ID")
    parser.add_argument("--pair-id", type=int, default=0, help="ID to generate default client/target pairs")
    
    # Environment Arguments
    parser.add_argument("--battle-format", default="1v1", choices=["1v1", "3v3"], help="Battle format")
    parser.add_argument("--observation-mode", default="minimal", choices=["minimal", "full"], help="State mode")
    
    # PPO Hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps to run for each environment per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epoch when optimizing the surrogate loss")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient for exploration")
    
    # Saving & Evaluation Config
    parser.add_argument("--save-freq", type=int, default=20_000, help="Save checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=0, help="Evaluate every X steps. Set to 0 to disable or use --n-evals instead.")
    parser.add_argument("--n-evals", type=int, default=5, help="Total number of evaluations during training. Overrides --eval-freq if > 0.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes to play during each evaluation.")
    parser.add_argument("--run-name", default="ppo_run", help="Name of the experiment run")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs" / "ppo"), help="Output directory")
    
    return parser

def resolve_ids(pair_id: int, client_id: str, target_id: str) -> tuple[str, str]:
    final_client = client_id if client_id else f"rl_{pair_id}"
    final_target = target_id if target_id else f"pvpoke_{pair_id}"
    return final_client, final_target

def make_env(ws_uri, client_id, target_id, format, obs_mode):
    """Instantiates the base environment and applies Monitor + ActionMasker."""
    base_env = PVPokeEnv(
        ws_uri, 
        client_id, 
        target_id, 
        battle_format=format, 
        observation_mode=obs_mode,
        reset_random=False
    )
    base_env.loop.run_until_complete(base_env.connect())
    # Notice we use ActionMasker here instead of our custom Dict wrapper
    return Monitor(ActionMasker(base_env, mask_fn))

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = build_arg_parser().parse_args()

    # Create output and logging folders
    model_dir = os.path.join(args.output_dir, args.run_name, "models")
    tb_dir = os.path.join(args.output_dir, args.run_name, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    client_id, target_id = resolve_ids(args.pair_id, args.client_id, args.target_id)

    logging.info(f"Starting Maskable PPO Training ({args.total_timesteps} steps)")
    logging.info(f"Connecting to {args.ws_uri} | client={client_id}, target={target_id}")

    # 1. Initialize ONE single environment connection
    env = make_env(args.ws_uri, client_id, target_id, args.battle_format, args.observation_mode)

    try:
        # 2. Build the MaskablePPO Model
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            tensorboard_log=tb_dir,
            policy_kwargs={"net_arch": [128, 64, 32]},
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=1,
        )

        # 3. Setup Callbacks
        callbacks = []
        
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=model_dir,
            name_prefix="ppo_ckpt",
        )
        callbacks.append(checkpoint_callback)

        # Calculate Evaluation Frequency
        eval_freq = args.eval_freq
        if args.n_evals > 0:
            eval_freq = args.total_timesteps // args.n_evals
            logging.info(f"Auto-calculated evaluation frequency: Every {eval_freq} steps (Total: {args.n_evals} evaluations).")

        # Inject Intermediate Evaluation if frequency > 0
        if eval_freq > 0:
            # We MUST use MaskableEvalCallback for sb3_contrib
            eval_callback = MaskableEvalCallback(
                env, 
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

        # 4. Train the Agent in one continuous run
        start_time = time.time()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        duration = time.time() - start_time

        # 5. Save Final Weights
        final_path = os.path.join(model_dir, "ppo_final_model")
        model.save(final_path)
        logging.info(f"Training finished in {duration/60:.2f} minutes. Model saved to {final_path}")

        # 6. Post-Training Evaluation (Using sb3_contrib's maskable evaluate_policy)
        logging.info("Starting post-training evaluation (30 episodes)...")
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=30, 
            deterministic=True
        )
        logging.info(f"Final Evaluation Results: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    finally:
        # Close WebSocket connection cleanly
        env.close()
        logging.info("Environment connection closed safely.")

if __name__ == "__main__":
    main()
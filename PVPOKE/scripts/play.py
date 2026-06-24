"""
PvPoke Reinforcement Learning Inference and Evaluation Pipeline.

This script loads a trained Reinforcement Learning agent (PPO, DQN, or A2C) 
and evaluates its performance against the PvPoke simulator in a deterministic 
manner to calculate its final win rate.

Usage:
    python play.py --algo ppo --model-path models/best_model.zip --episodes 10
"""

import argparse
import logging
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ── PATH CONFIGURATION ────────────────────────────────────────────────────────
# Resolve repository root directory and append it to sys.path to manage internal imports cleanly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Internal environment and custom RL components
from environment.ClassPVPOKE import PVPokeEnv
from stable_baselines3 import A2C
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from rl.masked_dqn import MaskedDQN


# ── ENVIRONMENT WRAPPERS ──────────────────────────────────────────────────────

class MaskedDictObservationWrapper(gym.Wrapper):
    """
    Observation wrapper for DQN and A2C algorithms.
    
    Transforms the standard environment state into a structured Dictionary 
    space containing both the raw observation tensor and its corresponding 
    valid action mask.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "obs": env.observation_space,
            "action_mask": spaces.Box(low=0.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32),
        })

    def _obs_with_mask(self, obs: np.ndarray) -> dict:
        """Helper method to package the current observation and action mask."""
        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "action_mask": np.asarray(self.env.action_masks(), dtype=np.float32),
        }

    def reset(self, **kwargs):
        """Resets the environment and returns the wrapped dictionary observation."""
        obs, info = self.env.reset(**kwargs)
        return self._obs_with_mask(obs), info

    def step(self, action: int):
        """Executes an environment step and wraps the resulting observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs_with_mask(obs), reward, terminated, truncated, info


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Retrieves the valid action mask from the core PvPoke environment.
    Required by the MaskablePPO architecture during inference.
    """
    return env.action_masks()


# ── CLI ARGUMENT PARSER ───────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Configures the command-line interface arguments for the evaluation pipeline.
    Includes sensible defaults for a plug-and-play production setup.
    """
    parser = argparse.ArgumentParser(
        description="Inference and evaluation script for trained PvPoke RL models."
    )
    
    # Network Connection Settings
    parser.add_argument("--ws-uri", default="ws://localhost:8000/ws", 
                        help="WebSocket URI for the PvPoke server bridge.")
    parser.add_argument("--client-id", default="rl_0", 
                        help="Unique identifier for the RL Agent client.")
    parser.add_argument("--target-id", default="pvpoke_0", 
                        help="Unique identifier for the target oponent.")
    parser.add_argument("--pair-id", type=int, default=0, 
                        help="Session ID used to pair matching clients.")
    
    # Simulation Environment Parameters
    parser.add_argument("--battle-format", default="1v1", choices=["1v1", "3v3"], 
                        help="Battle matchup framework.")
    parser.add_argument("--observation-mode", default="minimal", choices=["minimal", "full"], 
                        help="State space abstraction level.")
    parser.add_argument("--preprocess-mode", default="none", choices=["none", "normalized", "discrete"], 
                        help="Feature scaling strategy applied to inputs.")
    
    # Model Configuration
    parser.add_argument("--algo", type=str, default="ppo", choices=["dqn", "a2c", "ppo"], 
                        help="Algorithm architecture of the saved checkpoint.")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Absolute or relative path to the trained model (.zip) file.")
    
    # Evaluation Parameters
    parser.add_argument("--episodes", type=int, default=10, 
                        help="Total number of consecutive games to evaluate.")
    
    return parser


# ── MAIN PIPELINE EXECUTION ───────────────────────────────────────────────────

def main():
    # Setup structured console logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()

    logging.info(f"Starting inference session using architecture: {args.algo.upper()}")
    logging.info(f"Loading checkpoint weights from: {args.model_path}")

    # 1. Initialize the localized tracking environment
    base_env = PVPokeEnv(
        args.ws_uri, 
        args.client_id, 
        args.target_id, 
        battle_format=args.battle_format,         
        observation_mode=args.observation_mode,   
        reset_random=False
    )
    
    # Execute the asynchronous handshake protocol to establish the WebSocket channel
    base_env.loop.run_until_complete(base_env.connect())

    # 2. Apply algorithm-specific observation wrappers and load policy weights
    if args.algo == "ppo":
        env = ActionMasker(base_env, mask_fn)
        model = MaskablePPO.load(args.model_path)
    elif args.algo == "dqn":
        env = MaskedDictObservationWrapper(base_env)
        model = MaskedDQN.load(args.model_path)
    elif args.algo == "a2c":
        env = MaskedDictObservationWrapper(base_env)
        model = A2C.load(args.model_path)

    wins = 0

    try:
        # 3. Core Evaluation Loop (Pure inference, exploration noise disabled)
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            ep_reward = 0

            while not done:
                # Deterministic actions guarantee the agent selects the highest-value option
                if args.algo == "ppo":
                    # MaskablePPO requires action masks passed explicitly at prediction time
                    action_masks = env.action_masks()
                    action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
                else:
                    # DQN and A2C handle masks internally via the wrapped observation dictionary
                    action, _states = model.predict(obs, deterministic=True)
                
                # Advance the simulator state by executing the selected action
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated

            # Evaluation metrics track successful encounters via positive cumulative returns
            if ep_reward > 0:
                wins += 1
            
            logging.info(f"Episode {ep}/{args.episodes} complete | Cumulative Reward: {ep_reward}")

        # Calculate and print performance indicators
        winrate = (wins / args.episodes) * 100
        logging.info(" ── EVALUATION REPORT ── ")
        logging.info(f"Agent Win Rate: {winrate:.2f}% ({wins} victories out of {args.episodes} games)")

    finally:
        # Gracefully terminate the WebSocket socket interface connection
        env.close()
        logging.info("WebSocket bridge session disconnected cleanly.")


if __name__ == "__main__":
    main()
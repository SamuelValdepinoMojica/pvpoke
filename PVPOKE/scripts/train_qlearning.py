import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from rl.qlearning import QLearningConfig, run_qlearning_experiment


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train tabular Q-learning with optional action masking.",
    )
    parser.add_argument("--ws-uri", default="ws://localhost:8000/ws")
    parser.add_argument("--pair-id", type=int, default=0)
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--target-id", default=None)
    parser.add_argument("--battle-format", default="1v1", choices=["1v1", "3v3"])
    parser.add_argument("--observation-mode", default="minimal", choices=["minimal", "full"])
    parser.add_argument("--preprocess-mode", default="discrete", choices=["discrete", "normalized"])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--reset-random", action="store_true")
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.99)
    parser.add_argument("--run-prefix", default="ql")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs" / "qlearning"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wait-for-client", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def resolve_ids(pair_id: int, client_id: str, target_id: str) -> tuple[str, str]:
    if client_id:
        final_client = client_id
    else:
        final_client = f"rl_{pair_id}"

    if target_id:
        final_target = target_id
    else:
        final_target = f"pvpoke_{pair_id}"

    return final_client, final_target


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()

    client_id, target_id = resolve_ids(args.pair_id, args.client_id, args.target_id)
    use_mask = True
    if args.no_mask:
        use_mask = False
    elif args.use_mask:
        use_mask = True

    logging.info("Python client_id=%s | target_id=%s", client_id, target_id)
    logging.info("JS client should use client_id=%s | target_id=%s", target_id, client_id)
    logging.info("WS URI: %s", args.ws_uri)

    if args.wait_for_client:
        input("Press Enter when the JS client is connected...")

    config = QLearningConfig(
        ws_uri=args.ws_uri,
        client_id=client_id,
        target_id=target_id,
        battle_format=args.battle_format,
        observation_mode=args.observation_mode,
        preprocess_mode=args.preprocess_mode,
        discretization_bins=args.bins,
        reset_random=args.reset_random,
        use_mask=use_mask,
        n_training_episodes=args.train_episodes,
        eval_interval=args.eval_interval,
        n_eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        run_prefix=args.run_prefix,
        run_name=args.run_name,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        show_progress=not args.no_progress,
    )

    summary = run_qlearning_experiment(config)

    logging.info(
        "Done | win_rate=%.2f | mean_reward=%.3f | coverage=%.2f%%",
        summary.get("final_eval_win_rate", 0.0),
        summary.get("final_eval_mean_reward", 0.0),
        summary.get("coverage", 0.0),
    )


if __name__ == "__main__":
    main()

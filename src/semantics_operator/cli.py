import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reversible weight-space semantic experiments")
    parser.add_argument("command", choices=("inspect", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.toml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="", help="Named model profile in TOML")
    parser.add_argument("--steps", type=int, help="Override steps per primitive operator")
    args = parser.parse_args(argv)
    try:
        cfg = load_settings(args.config, args.env_file, args.model)
        if args.steps is not None:
            cfg = replace(cfg, steps=args.steps)
        from .model import LocalLanguageModel

        lm = LocalLanguageModel.load(cfg)
        if args.command == "inspect":
            print(f"Profile: {cfg.profile}; device: {lm.device}")
            for name, shape in lm.linear_modules().items():
                print(f"{name}\t{shape}")
            print("Select an internal Linear module in the model profile target_module field.")
            return 0
        from .artifacts import save_run
        from .experiment import run_experiment

        report, patches, representations = run_experiment(lm, cfg, progress=print)
        folder = save_run(cfg.output_dir, report, patches, representations)
        print(f"Results: {folder}")
        print("Inspect changed/unchanged metrics and controls; this run does not prove reasoning.")
        return 0
    except (ValueError, OSError, RuntimeError, FloatingPointError) as error:
        print(f"semop: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

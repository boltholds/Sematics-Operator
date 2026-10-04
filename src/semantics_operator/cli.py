import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reversible weight-space semantic experiments")
    parser.add_argument(
        "command",
        choices=(
            "inspect",
            "run",
            "steer",
            "compare",
            "localize",
            "reft",
            "das",
            "diagnose",
            "transfer",
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.toml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="", help="Named model profile in TOML")
    parser.add_argument("--steps", type=int, help="Override steps per primitive operator")
    parser.add_argument("--rank", type=int, help="Low-rank dimension for run/reft/das/transfer")
    parser.add_argument("--learning-rate", type=float, help="Training learning rate")
    parser.add_argument("--locality-weight", type=float, help="Training locality loss weight")
    parser.add_argument(
        "--seed", type=int, help="Override experiment and operator initialization seed"
    )
    parser.add_argument(
        "--train-representation",
        choices=("en", "ru", "symbolic"),
        default="en",
        help="transfer: sole training/validation representation; all three are evaluated",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Greedy token budget for diagnose/reft/transfer",
    )
    parser.add_argument(
        "--intervention-audit",
        action="store_true",
        help="diagnose: compare legacy force instructions with equation replacement",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("full_vocab", "binary"),
        default="full_vocab",
        help="reft/transfer: digit+EOS full-vocabulary loss (default) or binary-candidate ablation",
    )
    parser.add_argument(
        "--train-schemes",
        nargs="+",
        choices=("and_copy", "and_inverted"),
        default=["and_copy", "and_inverted"],
        help="reft: training and validation mechanisms",
    )
    parser.add_argument(
        "--reft-position",
        choices=("state", "answer"),
        default="state",
        help="reft/transfer: edit the shared state before the question (default), or the answer position on identical prompts",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        help="MLP paths; compare: indices/paths; reft: candidate blocks; das: one block (default 12); transfer: one fixed block (default 8)",
    )
    parser.add_argument(
        "--site-kind", choices=("mlp", "block"), default="mlp", help="compare injection site"
    )
    parser.add_argument(
        "--boundary",
        choices=("prompt", "decision"),
        default="prompt",
        help="compare position: last prompt or shared answer prefix",
    )
    parser.add_argument(
        "--preservation-weight",
        type=float,
        default=None,
        help="Validation damage penalty (transfer default 1, others 0): all unaffected nodes for reft/transfer; source/switch/flag for compare",
    )
    parser.add_argument(
        "--strengths", nargs="+", type=float, help="Validation steering grid; zero always included"
    )
    parser.add_argument(
        "--ridge", type=float, default=0.1, help="Positive regularization for compare"
    )
    parser.add_argument(
        "--pca-components",
        nargs="+",
        type=int,
        default=[],
        help="Train-only PCA ranks for compare/reft; enables compare exact-donor diagnostics",
    )
    parser.add_argument(
        "--layer-sets", nargs="+", help="Layer index sets for localize, e.g. 8 6,7,8 2,5,8"
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, help="Last aligned token counts for localize"
    )
    parser.add_argument(
        "--boundaries",
        nargs="+",
        choices=("prompt", "decision"),
        help="Patch before or after common answer prefix",
    )
    args = parser.parse_args(argv)
    preservation_weight = args.preservation_weight
    if preservation_weight is None:
        preservation_weight = 1.0 if args.command == "transfer" else 0.0
    try:
        cfg = load_settings(args.config, args.env_file, args.model)
        if args.steps is not None:
            cfg = replace(cfg, steps=args.steps)
        for key in ("rank", "learning_rate", "locality_weight", "seed"):
            value = getattr(args, key)
            if value is not None:
                cfg = replace(cfg, **{key: value})
        from .model import LocalLanguageModel

        if args.command == "transfer" and args.layers and len(args.layers) != 1:
            raise ValueError("transfer requires one predeclared block, e.g. --layers 8")
        lm = LocalLanguageModel.load(cfg)
        if args.command == "transfer":
            from .transfer_experiment import run_transfer
            from .transfer_reporting import save_transfer

            report, tensors = run_transfer(
                lm,
                cfg,
                layer=int(args.layers[0]) if args.layers else 8,
                train_representation=args.train_representation,
                strengths=args.strengths,
                preservation_weight=preservation_weight,
                position=args.reft_position,
                max_new_tokens=args.max_new_tokens,
                loss_mode=args.loss_mode,
                progress=print,
            )
            print(f"Results: {save_transfer(cfg.output_dir, report, tensors)}")
            return 0
        if args.command == "diagnose":
            if args.intervention_audit:
                from .intervention_diagnostics import (
                    run_intervention_diagnostics,
                    save_intervention_diagnostics,
                )

                report = run_intervention_diagnostics(
                    lm, cfg, max_new_tokens=args.max_new_tokens, progress=print
                )
                print(f"Results: {save_intervention_diagnostics(cfg.output_dir, report)}")
                return 0
            from .diagnostics import run_diagnostics, save_diagnostics

            report = run_diagnostics(lm, cfg, max_new_tokens=args.max_new_tokens, progress=print)
            print(f"Results: {save_diagnostics(cfg.output_dir, report)}")
            return 0
        if args.command in ("reft", "das"):
            from .reft_experiment import run_reft_suite, save_research

            if args.command == "das" and args.layers and len(args.layers) != 1:
                raise ValueError("das requires one block index, e.g. --layers 12")
            layer = int(args.layers[0]) if args.layers else 12
            if args.command == "reft":
                report, tensors = run_reft_suite(
                    lm,
                    cfg,
                    layer=layer,
                    localization_layers=[int(i) for i in args.layers] if args.layers else None,
                    max_new_tokens=args.max_new_tokens,
                    strengths=args.strengths,
                    pca_components=args.pca_components or None,
                    preservation_weight=preservation_weight,
                    progress=print,
                    loss_mode=args.loss_mode,
                    train_schemes=args.train_schemes,
                    position=args.reft_position,
                )
            else:
                from .das_experiment import run_das

                report, tensors = run_das(lm, cfg, layer=layer, progress=print)
            print(f"Results: {save_research(cfg.output_dir, report, tensors)}")
            return 0
        if args.command == "inspect":
            print(f"Profile: {cfg.profile}; device: {lm.device}")
            for name, shape in lm.linear_modules().items():
                print(f"{name}\t{shape}")
            print("Select an internal Linear module in the model profile target_module field.")
            return 0
        if args.command == "localize":
            from .localization import run_localization, save_localization

            groups = (
                [tuple(int(i) for i in group.split(",")) for group in args.layer_sets]
                if args.layer_sets
                else None
            )
            report = run_localization(
                lm,
                cfg,
                layer_sets=groups,
                windows=args.windows,
                boundaries=args.boundaries,
                progress=print,
            )
            folder = save_localization(cfg.output_dir, report)
            print(f"Results: {folder}")
            return 0
        if args.command == "compare":
            from .conditional import run_conditional
            from .steering import save_steering

            report, tensors = run_conditional(
                lm,
                cfg,
                layers=args.layers,
                strengths=args.strengths,
                ridge=args.ridge,
                pca_components=args.pca_components,
                site_kind=args.site_kind,
                boundary=args.boundary,
                preservation_weight=preservation_weight,
                progress=print,
            )
            folder = save_steering(cfg.output_dir, report, tensors)
            print(f"Results: {folder}")
            return 0
        if args.command == "steer":
            from .steering import run_steering, save_steering

            report, vectors = run_steering(
                lm, cfg, layers=args.layers, strengths=args.strengths, progress=print
            )
            folder = save_steering(cfg.output_dir, report, vectors)
            print(f"Results: {folder}")
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

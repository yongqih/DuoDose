"""User-facing DuoDose command-line interface."""

from __future__ import annotations

import argparse
import logging

from . import __version__
from .models.registry import BACKEND_SPECS, DEFAULT_DUODOSE_BACKEND


def _cmd_info(_args: argparse.Namespace) -> None:
    print(f"DuoDose {__version__}")
    print(f"Default backend: {DEFAULT_DUODOSE_BACKEND}")
    print("Available backends:")
    for alias, spec in BACKEND_SPECS.items():
        print(f"  {alias}: {spec.display_name} ({spec.role})")
    try:
        import torch

        print("PyTorch available: yes")
        cuda = bool(torch.cuda.is_available())
        print(f"CUDA available: {'yes' if cuda else 'no'}")
        if cuda:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch available: no")
        print("CUDA available: no")


def _cmd_run(args: argparse.Namespace) -> None:
    try:
        from anndata import read_h5ad
    except ImportError as exc:
        raise SystemExit("anndata is required for 'duodose run'") from exc
    from .api import DuoDose

    adata = read_h5ad(args.input)
    detector = DuoDose(
        backend=args.backend,
        expected_doublet_rate=args.expected_doublet_rate,
        random_state=args.seed,
        device=args.device,
        layer=args.layer,
        threshold_strategy=args.threshold_strategy,
        threshold=args.threshold,
        training_preset=args.training_preset,
        amp=args.amp,
        dl_batch_size=args.dl_batch_size,
        dl_max_epochs=args.dl_max_epochs,
        dl_patience=args.dl_patience,
    )
    result = detector.fit_predict(adata)
    result.add_to_adata(adata)
    adata.write_h5ad(args.output)
    print(f"Wrote DuoDose results to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="duodose", description="Homotypic-aware doublet detection for scRNA-seq.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show installed backends and accelerator availability.")
    info.set_defaults(func=_cmd_info)

    run = subparsers.add_parser("run", help="Run DuoDose on an input h5ad file.")
    run.add_argument("input", help="Input .h5ad file.")
    run.add_argument("--output", required=True, help="Output DuoDose-annotated .h5ad file.")
    run.add_argument("--layer", default=None, help="Raw-count layer. Uses adata.X when omitted.")
    run.add_argument("--backend", choices=list(BACKEND_SPECS), default=DEFAULT_DUODOSE_BACKEND)
    run.add_argument("--expected-doublet-rate", type=float, default=0.08)
    run.add_argument("--threshold-strategy", choices=["expected_rate", "probability", "none"], default="expected_rate")
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--training-preset", choices=["fast", "default", "robust"], default="default")
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--amp", action="store_true")
    run.add_argument("--dl-batch-size", type=int, default=None)
    run.add_argument("--dl-max-epochs", type=int, default=200)
    run.add_argument("--dl-patience", type=int, default=20)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--verbose", action="store_true", help="Enable informational progress logging.")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING)
    if getattr(args, "threshold_strategy", None) == "none":
        args.threshold_strategy = None
    args.func(args)


if __name__ == "__main__":
    main()

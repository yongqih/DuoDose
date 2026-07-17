import pytest

from duodose.cli import build_parser


def test_cli_exposes_only_frozen_backends() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "input.h5ad", "--output", "output.h5ad"])
    assert args.backend == "rf"
    for backend in ("rf", "dl"):
        parsed = parser.parse_args(["run", "input.h5ad", "--output", "output.h5ad", "--backend", backend])
        assert parsed.backend == backend
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "input.h5ad", "--output", "output.h5ad", "--backend", "logistic"])

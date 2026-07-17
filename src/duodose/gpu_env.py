"""Small GPU/CUDA environment report helpers for DuoDose DL runs."""

from __future__ import annotations

from pathlib import Path
import platform
import sys


def _recommended_batch_size(total_memory_bytes: int | None) -> int:
    if not total_memory_bytes:
        return 256
    gib = float(total_memory_bytes) / float(1024**3)
    if gib >= 24:
        return 8192
    if gib >= 12:
        return 4096
    if gib >= 8:
        return 2048
    return 1024


def build_gpu_env_report() -> dict[str, object]:
    report: dict[str, object] = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_available": False,
        "torch_version": "",
        "torch_cuda_is_available": False,
        "torch_cuda_version": "",
        "gpu_name": "",
        "total_gpu_memory_bytes": 0,
        "total_gpu_memory_gib": 0.0,
        "current_device": "",
        "amp_available": False,
        "cudnn_enabled": False,
        "recommended_batch_size_guess": 256,
        "warning": "PyTorch is not installed; DL training will fall back to CPU or be skipped.",
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        report["warning"] = f"PyTorch import failed: {exc}; DL training will fall back to CPU or be skipped."
        return report

    cuda_available = bool(torch.cuda.is_available())
    report.update(
        {
            "torch_available": True,
            "torch_version": str(getattr(torch, "__version__", "")),
            "torch_cuda_is_available": cuda_available,
            "torch_cuda_version": str(getattr(torch.version, "cuda", "")),
            "cudnn_enabled": bool(getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.enabled),
            "warning": "" if cuda_available else "WARNING: torch CUDA is not available; DuoDose DL training will fall back to CPU.",
        }
    )
    if cuda_available:  # pragma: no cover - depends on local hardware
        device_index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device_index)
        memory_bytes = int(getattr(props, "total_memory", 0))
        report.update(
            {
                "gpu_name": str(torch.cuda.get_device_name(device_index)),
                "total_gpu_memory_bytes": memory_bytes,
                "total_gpu_memory_gib": round(float(memory_bytes) / float(1024**3), 3),
                "current_device": str(device_index),
                "amp_available": True,
                "recommended_batch_size_guess": _recommended_batch_size(memory_bytes),
            }
        )
    return report


def format_gpu_env_report(report: dict[str, object]) -> str:
    ordered = [
        "python_version",
        "python_executable",
        "platform",
        "torch_available",
        "torch_version",
        "torch_cuda_is_available",
        "torch_cuda_version",
        "gpu_name",
        "total_gpu_memory_bytes",
        "total_gpu_memory_gib",
        "current_device",
        "amp_available",
        "cudnn_enabled",
        "recommended_batch_size_guess",
        "warning",
    ]
    return "\n".join(f"{key}: {report.get(key, '')}" for key in ordered) + "\n"


def write_gpu_env_report(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = format_gpu_env_report(build_gpu_env_report())
    output.write_text(text, encoding="utf-8")
    return output

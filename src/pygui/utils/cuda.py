"""CUDA detection and GPU utilities."""


def is_cuda_available() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import cupy as cp  # type: ignore

        cp.cuda.Device(0).use()
        return True
    except Exception:
        return False


def get_cuda_version() -> str | None:
    """Get CUDA version.

    Returns:
        CUDA version string or None if not available
    """
    try:
        import cupy as cp  # type: ignore

        return cp.cuda.runtime.runtimeGetVersion()
    except Exception:
        return None


def get_gpu_info() -> dict:
    """Get GPU information.

    Returns:
        Dictionary with GPU details or empty dict if CUDA not available
    """
    if not is_cuda_available():
        return {}

    try:
        import cupy as cp  # type: ignore

        device = cp.cuda.Device()
        return {
            "name": device.name,
            "compute_capability": device.compute_capability,
            "memory_free": device.mem_info[0],
            "memory_total": device.mem_info[1],
        }
    except Exception:
        return {}


def ensure_gpu_available() -> None:
    """Raise error if GPU is not available.

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not is_cuda_available():
        raise RuntimeError(
            "CUDA is not available. Please install cupy-cuda12x or use CPU-only mode."
        )


def get_device() -> str:
    """Get device to use (cpu or cuda).

    Returns:
        'cuda' if available, 'cpu' otherwise
    """
    return "cuda" if is_cuda_available() else "cpu"

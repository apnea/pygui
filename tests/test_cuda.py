"""Tests for CUDA utilities."""


from pygui.utils.cuda import (
    get_cuda_version,
    get_device,
    get_gpu_info,
    is_cuda_available,
)


def test_is_cuda_available():
    """Test CUDA availability check."""
    result = is_cuda_available()
    assert isinstance(result, bool)


def test_get_cuda_version():
    """Test getting CUDA version."""
    if is_cuda_available():
        version = get_cuda_version()
        assert version is not None
    else:
        version = get_cuda_version()
        assert version is None


def test_get_gpu_info():
    """Test getting GPU information."""
    info = get_gpu_info()

    if is_cuda_available():
        assert isinstance(info, dict)
        assert "name" in info
        assert "compute_capability" in info
        assert "memory_total" in info
        assert "memory_free" in info
    else:
        assert info == {}


def test_get_device():
    """Test getting device type."""
    device = get_device()
    assert device in ["cpu", "cuda"]

"""Optional CUDA C backend; importing this package requires neither CuPy nor CUDA."""
from .runtime import backend, current_config, dispatch_stats

__all__ = ["backend", "current_config", "dispatch_stats"]

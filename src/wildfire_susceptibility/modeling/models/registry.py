"""Re-exports the shared MODELS registry so callers can do
`from wildfire_susceptibility.modeling.models.registry import MODELS`
without needing to know it actually lives in core.registry."""

from ...core.registry import MODELS

__all__ = ["MODELS"]

from __future__ import annotations

from typing import Any, Dict, Optional


class BaseInternist:
    """Base class for runtime inspectors (internists).

    Responsibilities:
    - parse minimal config for enabling/interval
    - provide a small API for init/accumulate/finalize buffers
    - support hierarchical config: cfg.internist.<internist_name>
    """

    def __init__(self, cfg: Any, internist_name: str):
        """Initialize internist with hierarchical config.

        Args:
            cfg: Root config object
            internist_name: Name of this internist (e.g., 'activation_internist')
        """
        self.internist_name = internist_name

        # Parse internist-specific config from cfg.internist.<internist_name>
        self.internist_cfg = cfg

    def enabled(self) -> bool:
        """Check if this internist is enabled."""
        if self.internist_cfg is None:
            return False

        try:
            if hasattr(self.internist_cfg, "enabled"):
                return bool(self.internist_cfg.enabled)
            elif hasattr(self.internist_cfg, "get"):
                return bool(self.internist_cfg.get("enabled", False))
        except Exception:
            pass

        return False

    def get_interval(self) -> Optional[int]:
        """Get logging interval from config."""
        if self.internist_cfg is None:
            return None

        try:
            if hasattr(self.internist_cfg, "interval"):
                interval = int(self.internist_cfg.interval)
            else:
                interval = int(self.internist_cfg.get("interval", 1))
            return interval if interval > 0 else None
        except Exception:
            pass

        return None

    def _create_buffer(self) -> Dict[str, Any]:
        """Create a new buffer for this internist. To be implemented by subclasses."""
        raise NotImplementedError()

    def reset_buffer(self) -> None:
        """Reset internal buffer (called at the start of each training step)."""
        self._buffer = self._create_buffer()

    def accumulate(self, *args, **kwargs):
        raise NotImplementedError()

    def finalize(self, dp_allreduce) -> Dict[str, float]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.enabled()}, interval={self.get_interval()})"


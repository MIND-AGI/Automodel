from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


class DataScheduler(ABC):
    """Interface for data-mix schedulers used by BlendedJSONLDataset."""

    @abstractmethod
    def bind_sources(self, source_names: Sequence[str]) -> None:
        """Bind scheduler to source ordering used by the dataset."""

    @abstractmethod
    def get_weights(self, step: Optional[int] = None) -> List[float]:
        """Return normalized source probabilities for the given step or current scheduler step."""

    @abstractmethod
    def set_step(self, step: int) -> None:
        """Set the scheduler's current step."""

    @abstractmethod
    def get_step(self) -> int:
        """Get the scheduler's current step."""

    @abstractmethod
    def step(self, increment: int = 1) -> List[float]:
        """Advance the scheduler by increment steps and return current weights."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler checkpoint state."""

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler checkpoint state."""


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    values = [float(w) for w in weights]
    if any(w < 0 for w in values):
        raise ValueError(f"All scheduler weights must be >= 0, got {values}")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"Scheduler weights must sum to > 0, got {values}")
    return [w / total for w in values]


class UniformDataScheduler(DataScheduler):
    """Always returns uniform weights across all sources."""

    def __init__(self) -> None:
        self._source_names: List[str] = []
        self._current_step: int = 0

    def bind_sources(self, source_names: Sequence[str]) -> None:
        if len(source_names) == 0:
            raise ValueError("At least one source is required")
        self._source_names = list(source_names)

    def get_weights(self, step: Optional[int] = None) -> List[float]:
        if step is not None:
            self._current_step = int(step)
        if not self._source_names:
            raise RuntimeError("Data scheduler is not bound to sources")
        uniform = 1.0 / len(self._source_names)
        return [uniform] * len(self._source_names)

    def set_step(self, step: int) -> None:
        self._current_step = int(step)

    def get_step(self) -> int:
        return self._current_step

    def step(self, increment: int = 1) -> List[float]:
        self._current_step += int(increment)
        return self.get_weights()

    def state_dict(self) -> Dict[str, Any]:
        return {"current_step": self._current_step}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return None
        self._current_step = int(state.get("current_step", 0))
        return None


class StepwiseDataScheduler(DataScheduler):
    """
    Data scheduler with const or step-wise changing weights.

    Supports loading config from either direct arguments or a YAML config path.
    """

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        weights: Optional[Sequence[float] | Dict[str, float]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._source_names: List[str] = []
        self._current_step: int = 0
        self._base_weights_input = weights
        self._scheduler_input = scheduler
        self._step_weights: List[Tuple[int, List[float]]] = []

        if config_path is not None:
            with open(Path(config_path), "r") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError("data scheduler config YAML must be a mapping")
            self._base_weights_input = cfg.get("weights", self._base_weights_input)
            self._scheduler_input = cfg.get("scheduler", self._scheduler_input)

    def bind_sources(self, source_names: Sequence[str]) -> None:
        if len(source_names) == 0:
            raise ValueError("At least one source is required")
        self._source_names = list(source_names)
        base_weights = self._resolve_weights_input(self._base_weights_input, expected_names=self._source_names)
        if base_weights is None:
            uniform = 1.0 / len(self._source_names)
            base_weights = [uniform] * len(self._source_names)

        scheduler = self._scheduler_input
        if scheduler is None:
            self._step_weights = [(0, base_weights)]
            return

        if not isinstance(scheduler, dict):
            raise ValueError("scheduler must be a mapping")

        scheduler_type = str(scheduler.get("type", "const")).lower()
        if scheduler_type == "const":
            self._step_weights = [(0, base_weights)]
            return

        if scheduler_type in {"step_weights", "step_list"}:
            entries_cfg = scheduler.get("step_weights", scheduler.get("weights"))
            if entries_cfg is None:
                raise ValueError("step scheduler requires 'step_weights' or 'weights'")
            entries = self._parse_step_entries(entries_cfg, expected_names=self._source_names)
            if entries[0][0] > 0:
                # Before first milestone, use base weights.
                entries = [(0, base_weights)] + entries
            self._step_weights = entries
            return

        raise ValueError(
            "Unsupported data scheduler type. Supported values are 'const' and 'step_weights'"
        )

    def get_weights(self, step: Optional[int] = None) -> List[float]:
        if step is not None:
            self._current_step = int(step)
        if not self._source_names:
            raise RuntimeError("Data scheduler is not bound to sources")
        if not self._step_weights:
            raise RuntimeError("Data scheduler has no configured weights")

        effective_step = self._current_step
        current = self._step_weights[0][1]
        for scheduled_step, weights in self._step_weights:
            if effective_step < scheduled_step:
                break
            current = weights
        return current

    def set_step(self, step: int) -> None:
        self._current_step = int(step)

    def get_step(self) -> int:
        return self._current_step

    def step(self, increment: int = 1) -> List[float]:
        self._current_step += int(increment)
        return self.get_weights()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self._current_step,
            "step_weights": [
                {"step": step, "weights": weights}
                for step, weights in self._step_weights
            ]
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self._current_step = int(state.get("current_step", 0))
        entries = state.get("step_weights")
        if entries is None:
            return
        parsed: List[Tuple[int, List[float]]] = []
        for item in entries:
            parsed.append((int(item["step"]), _normalize_weights(item["weights"])))
        parsed.sort(key=lambda x: x[0])
        self._step_weights = parsed

    def _resolve_weights_input(
        self,
        weights_cfg: Optional[Sequence[float] | Dict[str, float]],
        *,
        expected_names: Sequence[str],
    ) -> Optional[List[float]]:
        if weights_cfg is None:
            return None

        if isinstance(weights_cfg, dict):
            missing = [name for name in expected_names if name not in weights_cfg]
            if missing:
                raise ValueError(
                    f"Missing source weights for names {missing}. Expected names={list(expected_names)}"
                )
            values = [float(weights_cfg[name]) for name in expected_names]
            return _normalize_weights(values)

        values = [float(w) for w in weights_cfg]
        if len(values) != len(expected_names):
            raise ValueError(
                f"weights length mismatch: expected {len(expected_names)}, got {len(values)}"
            )
        return _normalize_weights(values)

    def _parse_step_entries(
        self,
        cfg: Any,
        *,
        expected_names: Sequence[str],
    ) -> List[Tuple[int, List[float]]]:
        entries: List[Tuple[int, List[float]]] = []

        if isinstance(cfg, dict):
            for step_key, weights in cfg.items():
                step = int(step_key)
                norm = self._resolve_weights_input(weights, expected_names=expected_names)
                assert norm is not None
                entries.append((step, norm))
        elif isinstance(cfg, list):
            if not cfg:
                raise ValueError("step_weights cannot be empty")
            if all(isinstance(item, dict) and "step" in item and "weights" in item for item in cfg):
                for item in cfg:
                    step = int(item["step"])
                    norm = self._resolve_weights_input(item["weights"], expected_names=expected_names)
                    assert norm is not None
                    entries.append((step, norm))
            else:
                for step, weights in enumerate(cfg):
                    norm = self._resolve_weights_input(weights, expected_names=expected_names)
                    assert norm is not None
                    entries.append((step, norm))
        else:
            raise ValueError("step_weights must be a mapping or list")

        entries.sort(key=lambda x: x[0])
        if entries[0][0] < 0:
            raise ValueError("step values must be >= 0")
        deduped: List[Tuple[int, List[float]]] = []
        seen_steps = set()
        for step, weights in entries:
            if step in seen_steps:
                raise ValueError(f"Duplicate step value in scheduler: {step}")
            seen_steps.add(step)
            deduped.append((step, weights))
        return deduped

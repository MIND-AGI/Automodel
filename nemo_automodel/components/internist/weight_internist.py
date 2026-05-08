from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base_internist import BaseInternist


class WeightInternist(BaseInternist):
    """Internist that inspects model weight norms (embedding and lm_head).

    Reads config from:
      cfg.internist.weight_internist.enabled (bool)
      cfg.internist.weight_internist.interval (int, default 1)

    API:
      - enabled() -> bool
      - get_interval() -> Optional[int]
      - init_buffer() -> buffer dict
      - accumulate(model, buffer)
      - finalize(buffer, dp_allreduce) -> metrics dict
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg, internist_name="weight_internist")

    def _create_buffer(self) -> Dict[str, Any]:
        """Create a new buffer for weight norm accumulation."""
        return {
            "embedding_sum": None,
            "lm_head_sum": None,
            # gradient accumulators
            "embedding_grad_sum": None,
            "lm_head_grad_sum": None,
        }

    def accumulate(self, model: Any):
        """Accumulate weight norms directly from model parameters.

        Args:
            model: The model instance to extract embedding and lm_head weights from.
        """
        if model is None:
            return

        # Extract embedding weight directly from model
        embedding_weight = None
        if hasattr(model, "model") and hasattr(model.model, "embeddings"):
            embedding_weight = model.model.embeddings.weight
        elif hasattr(model, "embeddings"):
            embedding_weight = model.embeddings.weight

        # Process embedding weight
        if embedding_weight is not None:
            with torch.no_grad():
                # Accumulate local shard sum of squares; reduce + sqrt in finalize.
                local_sumsq = torch.sum(embedding_weight.to(torch.float32) ** 2)

                if self._buffer["embedding_sum"] is None:
                    self._buffer["embedding_sum"] = local_sumsq.detach()
                else:
                    self._buffer["embedding_sum"] += local_sumsq.detach()

            # Register backward hook to record gradient norm for embedding weight once
            try:
                if not getattr(embedding_weight, "_internist_hook_registered", False):
                    def _make_embedding_hook():
                        def _hook(grad):
                            if grad is None or self._buffer is None:
                                return
                            with torch.no_grad():
                                local_sumsq = torch.sum(grad.to(torch.float32) ** 2)
                                if self._buffer["embedding_grad_sum"] is None:
                                    self._buffer["embedding_grad_sum"] = local_sumsq.detach()
                                else:
                                    self._buffer["embedding_grad_sum"] += local_sumsq.detach()
                        return _hook

                    embedding_weight.register_hook(_make_embedding_hook())
                    setattr(embedding_weight, "_internist_hook_registered", True)
            except Exception:
                # Hook registration best-effort; continue without gradient recording if it fails
                pass

        # Extract lm_head weight directly from model
        lm_head_weight = None
        if hasattr(model, "lm_head"):
            lm_head_weight = model.lm_head.weight

        # Process lm_head weight
        if lm_head_weight is not None:
            with torch.no_grad():
                # Accumulate local shard sum of squares; reduce + sqrt in finalize.
                local_sumsq = torch.sum(lm_head_weight.to(torch.float32) ** 2)

                if self._buffer["lm_head_sum"] is None:
                    self._buffer["lm_head_sum"] = local_sumsq.detach()
                else:
                    self._buffer["lm_head_sum"] += local_sumsq.detach()

            # Register backward hook to record gradient norm for lm_head weight once
            try:
                if not getattr(lm_head_weight, "_internist_hook_registered", False):
                    def _make_lm_head_hook():
                        def _hook(grad):
                            if grad is None or self._buffer is None:
                                return
                            with torch.no_grad():
                                local_sumsq = torch.sum(grad.to(torch.float32) ** 2)
                                if self._buffer["lm_head_grad_sum"] is None:
                                    self._buffer["lm_head_grad_sum"] = local_sumsq.detach()
                                else:
                                    self._buffer["lm_head_grad_sum"] += local_sumsq.detach()
                        return _hook

                    lm_head_weight.register_hook(_make_lm_head_hook())
                    setattr(lm_head_weight, "_internist_hook_registered", True)
            except Exception:
                # Hook registration best-effort; continue without gradient recording if it fails
                pass

    def finalize(self, dp_allreduce) -> Dict[str, float]:
        """Finalize weight norm metrics and perform all-reduce.

        Args:
            dp_allreduce: Function to perform distributed all-reduce.

        Returns:
            Dict with metrics like 'weight_norm/embedding' and 'weight_norm/lm_head'.
        """
        if self._buffer is None:
            return {}

        metrics: Dict[str, float] = {}

        # Finalize embedding norm
        if self._buffer["embedding_sum"] is not None:
            reduced_sum = dp_allreduce(self._buffer["embedding_sum"], include_cp=True)
            metrics["weight_norm/embedding"] = torch.sqrt(reduced_sum)

        # Finalize embedding gradient norm
        if self._buffer.get("embedding_grad_sum") is not None:
            reduced_gsum = dp_allreduce(self._buffer["embedding_grad_sum"], include_cp=True)
            metrics["weight_grad_norm/embedding"] = torch.sqrt(reduced_gsum)

        # Finalize lm_head norm
        if self._buffer["lm_head_sum"] is not None:
            reduced_sum = dp_allreduce(self._buffer["lm_head_sum"], include_cp=True)
            metrics["weight_norm/lm_head"] = torch.sqrt(reduced_sum)

        # Finalize lm_head gradient norm
        if self._buffer.get("lm_head_grad_sum") is not None:
            reduced_gsum = dp_allreduce(self._buffer["lm_head_grad_sum"], include_cp=True)
            metrics["weight_grad_norm/lm_head"] = torch.sqrt(reduced_gsum)

        return metrics

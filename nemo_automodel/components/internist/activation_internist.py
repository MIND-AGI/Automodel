from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch

from .base_internist import BaseInternist


class ActivationInternist(BaseInternist):
    """Internist that inspects activations (hidden states) and accumulates norms.

    Reads config from:
      cfg.internist.activation_internist.enabled (bool)
      cfg.internist.activation_internist.interval (int, default 1)
      cfg.internist.activation_internist.enable_grad_norm (bool, default False)

    API:
      - enabled() -> bool
      - get_interval() -> Optional[int]
      - init_buffer() -> buffer dict
      - accumulate(hidden_states, buffer)
      - finalize(buffer, dp_allreduce) -> metrics dict
    """

    def __init__(self, cfg: Any, internist_name: str = "activation_internist"):
        super().__init__(cfg, internist_name=internist_name)
        # parse option: enable_grad_norm (default False)
        self.enable_grad_norm = bool(self.internist_cfg.get("enable_grad_norm", False))
        self._buffer = self._create_buffer()

    def __str__(self):
        return super().__str__() + f"(enable_grad_norm={self.enable_grad_norm})"

    def _create_buffer(self) -> Dict[str, list]:
        buf = {"sums": [], "counts": []}
        if self.enable_grad_norm:
            buf["grad_sums"] = []
        return buf

    def accumulate(self, hidden_states: Optional[Iterable[torch.Tensor]]):
        if self._buffer is None or hidden_states is None:
            return

        # Compute hidden_state norms per layer and accumulate sums and counts for mean calculation
        # hidden_states dimensions: list of [batch_size, seq_len, hidden_size]
        # norm on hidden_size dim, mean on batch and seq_len dims

        for layer_index, layer_hidden_states in enumerate(hidden_states):
            if layer_hidden_states is None:
                continue

            # compute stats without building gradients
            with torch.no_grad():
                layer_sum = torch.linalg.norm(layer_hidden_states, ord=2, dim=-1).sum()
            layer_count = torch.tensor(
                float(layer_hidden_states.shape[0] * layer_hidden_states.shape[1]),
                device=layer_hidden_states.device,
                dtype=torch.float32,
            )

            if layer_index >= len(self._buffer["sums"]):
                self._buffer["sums"].append(layer_sum)
                self._buffer["counts"].append(layer_count)
                if self.enable_grad_norm:
                    self._buffer["grad_sums"].append(torch.tensor(0.0, device=layer_hidden_states.device, dtype=torch.float32))
            else:
                self._buffer["sums"][layer_index] += layer_sum
                self._buffer["counts"][layer_index] += layer_count
                if self.enable_grad_norm and "grad_sums" in self._buffer:
                    # ensure grad_sums length
                    if layer_index >= len(self._buffer["grad_sums"]):
                        self._buffer["grad_sums"].append(torch.tensor(0.0, device=layer_hidden_states.device, dtype=torch.float32))

            # register hook to capture gradient norm for this activation if requested
            if self.enable_grad_norm:
                try:
                    if getattr(layer_hidden_states, "requires_grad", False):
                        def make_hook(idx):
                            def hook(grad):
                                with torch.no_grad():
                                    gsum = torch.linalg.norm(grad, ord=2, dim=-1).sum()
                                    # accumulate into buffer
                                    if idx >= len(self._buffer.get("grad_sums", [])):
                                        # extend if necessary
                                        self._buffer.setdefault("grad_sums", []).append(gsum.detach())
                                    else:
                                        self._buffer["grad_sums"][idx] += gsum.detach()
                            return hook

                        layer_hidden_states.register_hook(make_hook(layer_index))
                except Exception:
                    # non-critical: skip grad hooks if registration fails
                    pass

    def finalize(self, dp_allreduce) -> Dict[str, float]:
        if self._buffer is None or not self._buffer.get("sums"):
            return {}

        metrics: Dict[str, float] = {}
        grad_sums = self._buffer.get("grad_sums") if self.enable_grad_norm else None

        for layer_index, (layer_sum, layer_count) in enumerate(zip(self._buffer["sums"], self._buffer["counts"])):
            reduced_sum = dp_allreduce(layer_sum, include_cp=True)
            reduced_count = dp_allreduce(layer_count, include_cp=True)

            mean_hidden_state_norm = reduced_sum / max(reduced_count.item(), 1.0)
            metrics_name = f"activation_norm/layer_{layer_index}"
            if layer_index == 0:
                metrics_name = "activation_norm/embedding_layer"
            elif layer_index == len(self._buffer["sums"]) - 1:
                metrics_name = "activation_norm/final_layer"

            metrics[metrics_name] = mean_hidden_state_norm

            if grad_sums is not None and layer_index < len(grad_sums):
                reduced_gsum = dp_allreduce(grad_sums[layer_index], include_cp=True)
                metrics_name = f"activation_grad_norm/layer_{layer_index}"
                if layer_index == 0:
                    metrics_name = "activation_grad_norm/embedding_layer"
                elif layer_index == len(grad_sums) - 1:
                    metrics_name = "activation_grad_norm/final_layer"

                metrics[metrics_name] = reduced_gsum 

        return metrics


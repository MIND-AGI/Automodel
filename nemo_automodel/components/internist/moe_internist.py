from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base_internist import BaseInternist


class MoeInternist(BaseInternist):
    """Internist that inspects MoE router logits and computes expert scores.

    Reads config from:
      cfg.internist.moe_internist.enabled (bool)
      cfg.internist.moe_internist.interval (int, default 1)
      cfg.internist.moe_internist.aggregation (str, default "mean", options: "mean", "std", "min", "max")

    API:
      - enabled() -> bool
      - get_interval() -> Optional[int]
      - accumulate(router_logits) -> None
      - finalize(dp_allreduce) -> metrics dict
    """

    def __init__(self, cfg: Any, internist_name: str = "moe_internist"):
        super().__init__(cfg, internist_name=internist_name)
        # parse option: aggregation (default "mean")
        self.aggregation = str(self.internist_cfg.get("aggregation", "mean"))
        if self.aggregation not in ["mean", "std", "min", "max"]:
            self.aggregation = "mean"

    def __str__(self):
        return super().__str__() + f"(aggregation={self.aggregation})"

    def _create_buffer(self) -> Dict[str, list]:
        """Create buffer for accumulating expert scores per layer."""
        buf = {
            "layer_expert_scores": [],  # list of lists: [layer_idx][expert_scores_tensor]
            "num_layers": 0,
            "num_experts": 0,
        }
        return buf

    def accumulate(self, router_logits: Optional[list[torch.Tensor]]):
        """Accumulate router logits and compute expert scores per layer.

        Args:
            router_logits: List of router logits tensors, one per layer.
                          Each tensor has shape (batch_size * seq_len, num_experts).
        """
        if self._buffer is None or router_logits is None:
            return

        # Initialize on first call
        if self._buffer["num_layers"] == 0 and len(router_logits) > 0:
            self._buffer["num_layers"] = len(router_logits)
            self._buffer["num_experts"] = router_logits[0].shape[-1] if router_logits[0] is not None else 0
            self._buffer["layer_expert_scores"] = [[] for _ in range(len(router_logits))]

        # Process each layer's router logits
        for layer_idx, layer_router_logits in enumerate(router_logits):
            if layer_router_logits is None:
                continue

            if layer_idx >= len(self._buffer["layer_expert_scores"]):
                self._buffer["layer_expert_scores"].append([])

            # Compute softmax scores per expert
            # layer_router_logits shape: (batch_size * seq_len, num_experts)
            with torch.no_grad():
                scores = torch.softmax(layer_router_logits.to(torch.float32), dim=-1)
                # Compute mean score per expert across all tokens
                expert_mean_scores = scores.mean(dim=0)  # shape: (num_experts,)
                self._buffer["layer_expert_scores"][layer_idx].append(expert_mean_scores)

    def finalize(self, dp_allreduce) -> Dict[str, float]:
        """Finalize MoE expert score metrics.

        Args:
            dp_allreduce: Function to perform distributed all-reduce.

        Returns:
            Dict with metrics like 'moe_score/layer_{i}/expert_{j}'.
        """
        if self._buffer is None or not self._buffer.get("layer_expert_scores"):
            return {}

        metrics: Dict[str, float] = {}
        num_layers = self._buffer["num_layers"]
        num_experts = self._buffer["num_experts"]

        # Process each layer
        for layer_idx in range(num_layers):
            if layer_idx >= len(self._buffer["layer_expert_scores"]):
                continue

            layer_scores_list = self._buffer["layer_expert_scores"][layer_idx]
            if not layer_scores_list:
                continue

            # Stack all accumulated scores for this layer and aggregate
            # shape: (num_accumulations, num_experts)
            stacked_scores = torch.stack(layer_scores_list, dim=0)

            # Apply aggregation across accumulations
            if self.aggregation == "mean":
                aggregated = stacked_scores.mean(dim=0)
            elif self.aggregation == "std":
                aggregated = stacked_scores.std(dim=0)
            elif self.aggregation == "min":
                aggregated = stacked_scores.min(dim=0)[0]
            elif self.aggregation == "max":
                aggregated = stacked_scores.max(dim=0)[0]
            else:
                aggregated = stacked_scores.mean(dim=0)

            # All-reduce each expert score
            for expert_idx in range(num_experts):
                expert_score = aggregated[expert_idx]
                reduced_score = dp_allreduce(expert_score, include_cp=True)

                metric_name = f"moe_score/layer_{layer_idx}/expert_{expert_idx}"
                # Special naming for first and last layers
                if layer_idx == 0:
                    metric_name = f"moe_score/first_layer/expert_{expert_idx}"
                elif layer_idx == num_layers - 1:
                    metric_name = f"moe_score/final_layer/expert_{expert_idx}"

                metrics[metric_name] = reduced_score.item() if hasattr(reduced_score, "item") else float(reduced_score)

            # Also compute per-layer statistics
            layer_mean = aggregated.mean()
            reduced_layer_mean = dp_allreduce(layer_mean, include_cp=True)
            layer_metric_name = f"moe_score/layer_{layer_idx}/mean"
            if layer_idx == 0:
                layer_metric_name = "moe_score/first_layer/mean"
            elif layer_idx == num_layers - 1:
                layer_metric_name = "moe_score/final_layer/mean"
            metrics[layer_metric_name] = reduced_layer_mean.item() if hasattr(reduced_layer_mean, "item") else float(reduced_layer_mean)

            # Standard deviation across experts
            layer_std = aggregated.std()
            reduced_layer_std = dp_allreduce(layer_std, include_cp=True)
            layer_std_metric_name = f"moe_score/layer_{layer_idx}/std"
            if layer_idx == 0:
                layer_std_metric_name = "moe_score/first_layer/std"
            elif layer_idx == num_layers - 1:
                layer_std_metric_name = "moe_score/final_layer/std"
            metrics[layer_std_metric_name] = reduced_layer_std.item() if hasattr(reduced_layer_std, "item") else float(reduced_layer_std)

        return metrics

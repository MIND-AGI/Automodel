from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict

import yaml
from torch.utils.data import IterableDataset, get_worker_info

from nemo_automodel.components.datasets.llm.bufshuf_sjsonl_dataset import BufShufSJSONLDataset
from nemo_automodel.components.datasets.llm.data_scheduler import DataScheduler, UniformDataScheduler
from nemo_automodel.components.datasets.llm.sjsonl_dataset import SJSONLDataset


__all__ = [
    "BlendedJSONLDataset",
    "BlendedJSONLDatasetState",
]


class BlendedJSONLDatasetState(TypedDict):
    """Checkpoint state for weighted JSONL source mixing."""

    source_items: List[Tuple[str, str]]
    source_states: Dict[str, Dict[str, Any]]
    rng_state: dict
    samples_emitted: int
    global_worker_id: int
    total_workers: int
    dataset_type: str
    tokenizer_config: Dict[str, Any]
    data_scheduler_state: Dict[str, Any]


def _is_yaml_path(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    suffix = Path(value).suffix.lower()
    return suffix in {".yaml", ".yml"}


def _parse_source_list_entries(sources: Sequence[Any]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for idx, entry in enumerate(sources):
        if not isinstance(entry, dict):
            raise ValueError(
                "Source list entries must be dicts with keys 'name' and 'path'; "
                f"got {type(entry).__name__} at index {idx}"
            )
        if "path" not in entry or "name" not in entry:
            raise ValueError(
                "Each source entry must include 'name' and 'path' keys; "
                f"got {entry!r}"
            )
        items.append((str(entry["name"]), str(entry["path"])))
    return items


def _resolve_sources_input(
    sources: Dict[str, str] | Sequence[Any] | str | Path,
) -> Dict[str, str] | Sequence[Any]:
    if _is_yaml_path(sources):
        with open(Path(sources), "r") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Sources YAML must contain a top-level mapping")
        if "sources" not in cfg:
            raise ValueError("Sources YAML must contain a 'sources' field")
        return cfg["sources"]
    return sources


def _parse_sources(sources: Dict[str, str] | Sequence[Any]) -> List[Tuple[str, str]]:
    """
    Parse sources from either:
    1) dict[name, path_or_glob]
    2) list[dict] entries like [{"name": "a", "path": "..."}, ...]
    """
    if isinstance(sources, dict):
        items = [(str(name), str(path_or_glob)) for name, path_or_glob in sources.items()]
    else:
        items = _parse_source_list_entries(sources)

    if not items:
        raise ValueError("At least one source is required")

    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate source name found in sources config: {names}")

    return items


def _get_worker_id_and_total_workers(worker: Optional[Any]) -> tuple[int, int]:
    try:
        import torch.distributed as dist

        dist_world_size = dist.get_world_size() if dist.is_initialized() else 1
        dist_rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        dist_world_size = 1
        dist_rank = 0

    dl_num_workers = worker.num_workers if worker is not None else 1
    dl_worker_id = worker.id if worker is not None else 0

    total_workers = dist_world_size * dl_num_workers
    global_worker_id = dist_rank * dl_num_workers + dl_worker_id
    return global_worker_id, total_workers


class BlendedJSONLDataset(IterableDataset):
    """
    Weighted mixture over multiple SJSONL/BufShufSJSONL streaming datasets.

    This follows lingua's weighted source sampling idea while preserving
    worker-level checkpoint/resume determinism via StatefulDataLoader.
    """

    def __init__(
        self,
        sources: Dict[str, str] | Sequence[Any] | str | Path,
        seq_len: int,
        tokenizer: Any,
        tokenizer_config: Dict[str, Any],
        *,
        dataset_type: str = "bufshuf_sjsonl",
        shuffle_files: bool = False,
        text_key: str = "text",
        sample_buffer_size: int = 1000,
        prefetch_factor: int = 2,
        random_seed: int = 12345,
        data_scheduler: Optional[DataScheduler] = None,
        resume_state: Optional[BlendedJSONLDatasetState] = None,
    ) -> None:
        super().__init__()
        if dataset_type not in {"sjsonl", "bufshuf_sjsonl"}:
            raise ValueError(
                "dataset_type must be 'sjsonl' or 'bufshuf_sjsonl', "
                f"got {dataset_type}"
            )

        resolved_sources = _resolve_sources_input(sources)
        self.source_items = _parse_sources(resolved_sources)
        self.seq_len = int(seq_len)
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
        self.dataset_type = dataset_type
        self.shuffle_files = shuffle_files
        self.text_key = text_key
        self.sample_buffer_size = int(sample_buffer_size)
        self.prefetch_factor = int(prefetch_factor)
        self.random_seed = int(random_seed)
        self.resume_state = resume_state
        self.data_scheduler: DataScheduler = data_scheduler or UniformDataScheduler()
        self.data_scheduler.bind_sources([name for name, _ in self.source_items])

        self._scheduler_step: Optional[int] = None
        self._warned_internal_step_fallback = False

        self._current_state: Optional[BlendedJSONLDatasetState] = None
        self._source_datasets: Optional[Dict[str, IterableDataset]] = None

    def set_scheduler_step(self, step: int) -> None:
        """Set global train step used by data scheduler."""
        self._scheduler_step = int(step)
        self.data_scheduler.set_step(self._scheduler_step)

    def get_scheduler_step(self) -> Optional[int]:
        return self._scheduler_step

    def get_current_weights(self, step: Optional[int] = None) -> Dict[str, float]:
        names = [name for name, _ in self.source_items]
        if step is not None:
            self.data_scheduler.set_step(int(step))
        elif self._scheduler_step is not None:
            self.data_scheduler.set_step(int(self._scheduler_step))
        weights = self.data_scheduler.get_weights()
        return {name: float(weight) for name, weight in zip(names, weights)}

    def _set_current_state(
        self,
        *,
        source_items: List[Tuple[str, str]],
        source_states: Dict[str, Dict[str, Any]],
        rng_state: dict,
        samples_emitted: int,
        global_worker_id: int,
        total_workers: int,
    ) -> None:
        """Keep a live in-memory state and snapshot it only when checkpointing."""
        self._current_state = BlendedJSONLDatasetState(
            source_items=source_items,
            source_states=source_states,
            rng_state=rng_state,
            samples_emitted=samples_emitted,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
            dataset_type=self.dataset_type,
            tokenizer_config=self.tokenizer_config,
            data_scheduler_state=self.data_scheduler.state_dict(),
        )

    def state_dict(self) -> BlendedJSONLDatasetState:
        if self._current_state is None:
            worker = get_worker_info()
            global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)
            return BlendedJSONLDatasetState(
                source_items=self.source_items.copy(),
                source_states={},
                rng_state=random.Random(self.random_seed + global_worker_id).getstate(),
                samples_emitted=0,
                global_worker_id=global_worker_id,
                total_workers=total_workers,
                dataset_type=self.dataset_type,
                tokenizer_config=self.tokenizer_config,
                data_scheduler_state=self.data_scheduler.state_dict(),
            )
        # Get latest source states only when checkpointing
        state = copy.deepcopy(self._current_state)
        if self._source_datasets:
            for source_name, ds in self._source_datasets.items():
                if hasattr(ds, "state_dict"):
                    state["source_states"][source_name] = ds.state_dict()
        return state

    def load_state_dict(self, state: BlendedJSONLDatasetState) -> None:
        self.resume_state = state

    def _build_source_dataset(
        self,
        file_pattern: str,
        source_state: Optional[Dict[str, Any]],
    ) -> IterableDataset:
        if self.dataset_type == "sjsonl":
            return SJSONLDataset(
                file_pattern=file_pattern,
                seq_len=self.seq_len,
                tokenizer=self.tokenizer,
                tokenizer_config=self.tokenizer_config,
                shuffle_files=self.shuffle_files,
                text_key=self.text_key,
                resume_state=source_state,
            )
        
        elif self.dataset_type == "bufshuf_sjsonl":

            return BufShufSJSONLDataset(
                file_pattern=file_pattern,
                seq_len=self.seq_len,
                tokenizer=self.tokenizer,
                tokenizer_config=self.tokenizer_config,
                shuffle_files=self.shuffle_files,
                text_key=self.text_key,
                resume_state=source_state,
                sample_buffer_size=self.sample_buffer_size,
                prefetch_factor=self.prefetch_factor,
            )
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)

        if self.resume_state is not None:
            if self.resume_state["global_worker_id"] != global_worker_id:
                raise ValueError(
                    f"Resume state worker ID mismatch: expected {global_worker_id}, "
                    f"got {self.resume_state['global_worker_id']}"
                )
            if self.resume_state["total_workers"] != total_workers:
                raise ValueError(
                    f"Resume state total workers mismatch: expected {total_workers}, "
                    f"got {self.resume_state['total_workers']}"
                )
            source_items = self.resume_state["source_items"].copy()
            source_states = copy.deepcopy(self.resume_state["source_states"])
            samples_emitted = int(self.resume_state["samples_emitted"])
            rng = random.Random()
            rng.setstate(self.resume_state["rng_state"])
            scheduler_state = self.resume_state.get("data_scheduler_state", {})
            self.data_scheduler.load_state_dict(scheduler_state)

            # Backward compatibility for checkpoints that stored dataset-level scheduler_step.
            if isinstance(scheduler_state, dict) and "current_step" not in scheduler_state and "scheduler_step" in self.resume_state:
                self.data_scheduler.set_step(int(self.resume_state["scheduler_step"]))

            if self._scheduler_step is not None:
                self.data_scheduler.set_step(int(self._scheduler_step))
        else:
            source_items = self.source_items.copy()
            source_states = {}
            samples_emitted = 0
            rng = random.Random(self.random_seed + global_worker_id)
            if self._scheduler_step is not None:
                self.data_scheduler.set_step(int(self._scheduler_step))

        source_datasets: Dict[str, IterableDataset] = {}
        source_iterators: Dict[str, Iterator[dict]] = {}
        for source_name, path_or_glob in source_items:
            ds = self._build_source_dataset(path_or_glob, source_states.get(source_name))
            source_datasets[source_name] = ds
            source_iterators[source_name] = iter(ds)

            if source_name not in source_states and hasattr(ds, "state_dict"):
                source_states[source_name] = ds.state_dict()

        # Store reference to source_datasets for deferred state_dict() calls
        self._source_datasets = source_datasets

        self._set_current_state(
            source_items=source_items,
            source_states=source_states,
            rng_state=rng.getstate(),
            samples_emitted=samples_emitted,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
        )

        source_names = [item[0] for item in source_items]

        while True:
            effective_step: int
            if self._scheduler_step is not None:
                effective_step = int(self._scheduler_step)
                self.data_scheduler.set_step(effective_step)
            else:
                effective_step = int(self.data_scheduler.get_step())
                if not self._warned_internal_step_fallback:
                    logging.warning(
                        "BlendedJSONLDataset is using internal emitted-sample step as scheduler step. "
                        "For real train-step scheduling, call dataset.set_scheduler_step(step) from the training loop."
                    )
                    self._warned_internal_step_fallback = True

                self.data_scheduler.set_step(effective_step)

            probs = self.data_scheduler.get_weights()
            if len(probs) != len(source_names):
                raise ValueError(
                    f"Data scheduler returned {len(probs)} weights for {len(source_names)} sources"
                )

            selected_source = rng.choices(source_names, weights=probs, k=1)[0]
            sample = next(source_iterators[selected_source])
            samples_emitted += 1
            if self._scheduler_step is None:
                self.data_scheduler.step(1)

            self._set_current_state(
                source_items=source_items,
                source_states=source_states,
                rng_state=rng.getstate(),
                samples_emitted=samples_emitted,
                global_worker_id=global_worker_id,
                total_workers=total_workers,
            )

            yield sample

    def __len__(self) -> int:
        raise NotImplementedError("__len__ is not implemented for BlendedJSONLDataset.")

    def __getitem__(self, index: int):
        raise NotImplementedError("__getitem__ is not implemented for BlendedJSONLDataset.")
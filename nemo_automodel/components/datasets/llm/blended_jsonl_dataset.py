from __future__ import annotations

import copy
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict

from torch.utils.data import IterableDataset, get_worker_info

from nemo_automodel.components.datasets.llm.bufshuf_sjsonl_dataset import BufShufSJSONLDataset
from nemo_automodel.components.datasets.llm.sjsonl_dataset import SJSONLDataset


__all__ = [
    "BlendedJSONLDataset",
    "BlendedJSONLDatasetState",
]


class BlendedJSONLDatasetState(TypedDict):
    """Checkpoint state for weighted JSONL source mixing."""

    source_items: List[Tuple[str, float]]
    source_states: Dict[str, Dict[str, Any]]
    rng_state: dict
    samples_emitted: int
    global_worker_id: int
    total_workers: int
    dataset_type: str
    tokenizer_config: Dict[str, Any]


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _parse_weighted_sources(sources: Dict[str, float] | Sequence[str]) -> List[Tuple[str, float]]:
    """
    Parse weighted sources from either:
    1) dict[path_or_glob, weight]
    2) flattened list ["30", "/path/a*.jsonl", "70", "/path/b*.jsonl"]
    """
    if isinstance(sources, dict):
        items = [(str(path), float(weight)) for path, weight in sources.items()]
    else:
        if len(sources) == 0 or len(sources) % 2 != 0:
            raise ValueError("Flattened weighted sources must have even length: [w1, p1, w2, p2, ...]")
        items = []
        for idx in range(0, len(sources), 2):
            weight = sources[idx]
            path_or_glob = sources[idx + 1]
            if not _is_number(weight):
                raise ValueError(f"Expected numeric weight at index {idx}, got {weight!r}")
            items.append((str(path_or_glob), float(weight)))

    if not items:
        raise ValueError("At least one weighted source is required")

    total = 0.0
    for path_or_glob, weight in items:
        if weight <= 0:
            raise ValueError(f"Weight must be > 0 for source {path_or_glob}, got {weight}")
        total += weight

    if total <= 0:
        raise ValueError("Sum of weights must be > 0")

    return [(path_or_glob, weight / total) for path_or_glob, weight in items]


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
        sources: Dict[str, float] | Sequence[str],
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
        resume_state: Optional[BlendedJSONLDatasetState] = None,
    ) -> None:
        super().__init__()
        if dataset_type not in {"sjsonl", "bufshuf_sjsonl"}:
            raise ValueError(
                "dataset_type must be 'sjsonl' or 'bufshuf_sjsonl', "
                f"got {dataset_type}"
            )

        self.source_items = _parse_weighted_sources(sources)
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

        self._current_state: Optional[BlendedJSONLDatasetState] = None

    def _set_current_state(
        self,
        *,
        source_items: List[Tuple[str, float]],
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
            )
        return copy.deepcopy(self._current_state)

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
        else:
            source_items = self.source_items.copy()
            source_states = {}
            samples_emitted = 0
            rng = random.Random(self.random_seed + global_worker_id)

        source_datasets: Dict[str, IterableDataset] = {}
        source_iterators: Dict[str, Iterator[dict]] = {}
        for path_or_glob, _ in source_items:
            ds = self._build_source_dataset(path_or_glob, source_states.get(path_or_glob))
            source_datasets[path_or_glob] = ds
            source_iterators[path_or_glob] = iter(ds)

            if path_or_glob not in source_states and hasattr(ds, "state_dict"):
                source_states[path_or_glob] = ds.state_dict()

        self._set_current_state(
            source_items=source_items,
            source_states=source_states,
            rng_state=rng.getstate(),
            samples_emitted=samples_emitted,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
        )

        keys = [item[0] for item in source_items]
        probs = [item[1] for item in source_items]

        while True:
            selected_source = rng.choices(keys, weights=probs, k=1)[0]
            sample = next(source_iterators[selected_source])
            samples_emitted += 1

            selected_ds = source_datasets[selected_source]
            source_states[selected_source] = selected_ds.state_dict()

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
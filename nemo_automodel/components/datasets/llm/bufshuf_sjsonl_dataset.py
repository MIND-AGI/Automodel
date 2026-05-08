"""
BufShufSJSONLDataset with document-level buffering and shuffling.

This version solves loss fluctuation by:
1. Buffering multiple complete documents
2. Shuffling samples from different documents
3. Ensuring batches have diverse content (not all from same doc)

This approach is more effective than boundary tracking because it:
- Naturally mixes content from different documents
- Reduces correlation within batches
- Smooths out loss variance significantly
"""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path
from typing import Iterator, List, Sequence, Optional, TypedDict, Dict, Any

import torch
from torch.utils.data import IterableDataset, get_worker_info


__all__ = [
    "BufShufSJSONLDataset",
    "BufShufSJSONLDatasetState",
]


class JSONLFileState(TypedDict):
    """State for a single JSONL file being read."""
    file_path: str
    position: int
    line_number: int
    block_size: int
    offset: int
    current_iter: int


class BufShufBufferState(TypedDict):
    """State for the enhanced document buffer.
    
    Attributes:
        token_buffer: Accumulated tokens waiting to be sampled
        token_buffer_offset: Offset within token buffer
        sample_buffer: Pre-generated shuffled samples ready to yield
        sample_buffer_idx: Index of next sample to yield from sample_buffer
        documents_processed: Counter for documents seen
    """
    token_buffer: List[int]
    sample_buffer: List[Dict[str, List[int]]]
    sample_buffer_idx: int
    documents_processed: int


class BufShufSJSONLDatasetState(TypedDict):
    """Complete state for enhanced JSONL dataset checkpointing."""
    current_file_idx: int
    files_order: List[str]
    file_state: JSONLFileState
    buffer_state: BufShufBufferState
    rng_state: dict
    epoch: int
    global_worker_id: int
    total_workers: int
    tokenizer_config: Dict[str, Any]


def _get_worker_id_and_total_workers(worker: Optional[Any]) -> tuple[int, int]:
    """Get global worker ID and total workers across distributed training."""
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


class BufShufSJSONLDataset(IterableDataset):
    """
    Enhanced JSONL Dataset with document-level buffering and shuffling.
    
    Key improvements over base SJSONLDataset:
    1. **Document Buffer**: Accumulates samples from multiple documents
    2. **Sample Shuffling**: Shuffles samples before yielding to batch
    3. **Diversity**: Ensures batches contain samples from different documents
    4. **Reduced Correlation**: Adjacent batches are less correlated
    
    This significantly reduces loss fluctuation by preventing batches from
    being filled with consecutive samples from the same document.
    
    Args:
        file_pattern: Glob pattern or list of JSONL file paths
        seq_len: Length of sequences to produce
        tokenizer: Tokenizer instance with encode() method
        tokenizer_config: Dict with tokenizer configuration
        shuffle_files: Whether to shuffle file order each epoch
        text_key: Key in JSONL for text content
        resume_state: Optional state dict to resume from checkpoint
        sample_buffer_size: Number of samples to buffer before shuffling (default: 1000)
            Larger = more diversity but more memory
        prefetch_factor: How many samples to prefetch (default: 2)
    """

    def __init__(
        self,
        file_pattern: str | Sequence[str],
        seq_len: int,
        tokenizer: Any,
        tokenizer_config: Dict[str, Any],
        *,
        shuffle_files: bool = False,
        text_key: str = "text",
        resume_state: Optional[BufShufSJSONLDatasetState] = None,
        sample_buffer_size: int = 1000,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()
        
        # File discovery
        if isinstance(file_pattern, (str, Path)):
            self.files: List[str] = sorted(glob.glob(str(file_pattern)))
        else:
            self.files = list(map(str, file_pattern))
        
        if not self.files:
            raise FileNotFoundError(f"No files matched pattern {file_pattern}")
        
        self.seq_len = int(seq_len)
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
        self.shuffle_files = shuffle_files
        self.text_key = text_key
        self.resume_state = resume_state
        
        # Buffer configuration
        self.sample_buffer_size = sample_buffer_size
        self.prefetch_factor = prefetch_factor
        
        # Internal state tracking
        self._current_state: Optional[BufShufSJSONLDatasetState] = None

    def state_dict(self) -> BufShufSJSONLDatasetState:
        """Returns current state for checkpointing."""
        if self._current_state is None:
            worker = get_worker_info()
            global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)
            if len(self.files) >= total_workers:
                block_size, offset = 1, 0
            else:
                block_size, offset = total_workers, global_worker_id % total_workers

            return BufShufSJSONLDatasetState(
                current_file_idx=0,
                files_order=self.files.copy(),
                file_state=JSONLFileState(
                    file_path=self.files[0] if self.files else "",
                    position=0,
                    line_number=0,
                    block_size=block_size,
                    offset=offset,
                    current_iter=0,
                ),
                buffer_state=BufShufBufferState(
                    token_buffer=[],
                    sample_buffer=[],
                    sample_buffer_idx=0,
                    documents_processed=0,
                ),
                rng_state=random.Random().getstate(),
                epoch=0,
                global_worker_id=global_worker_id,
                total_workers=total_workers,
                tokenizer_config=self.tokenizer_config,
            )
        
        return self._current_state.copy()

    def load_state_dict(self, state: BufShufSJSONLDatasetState) -> None:
        """Load a previously saved state to resume iteration."""
        self.resume_state = state

    def _read_jsonl_lines(
        self,
        file_path: str,
        position: int,
        line_number: int,
        block_size: int,
        offset: int,
    ) -> Iterator[tuple[dict, int, int]]:
        """Read lines from JSONL file using file.seek() for resumption."""
        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(position)
            current_line = line_number
            
            while True:
                line = f.readline()
                if not line:
                    break
                
                if current_line % block_size == offset:
                    try:
                        data = json.loads(line)
                        next_position = f.tell()
                        next_line_number = current_line + 1
                        yield data, next_position, next_line_number
                    except json.JSONDecodeError:
                        pass
                
                current_line += 1

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text following official implementation."""
        add_bos = self.tokenizer_config.get("add_bos", False)
        add_eos = self.tokenizer_config.get("add_eos", False)
        max_length = self.tokenizer_config.get("max_length", None)
        
        tokens = self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=(max_length is not None),
        )
        
        if add_bos and tokens and tokens[0] != self.tokenizer.bos_token_id:
            tokens = [self.tokenizer.bos_token_id] + tokens
        
        if add_eos and tokens and tokens[-1] != self.tokenizer.eos_token_id:
            tokens = tokens + [self.tokenizer.eos_token_id]
        
        return tokens

    def _setup_worker_context(
        self,
        files: List[str],
        shuffle: bool,
        resume_state: Optional[BufShufSJSONLDatasetState],
    ) -> tuple[List[str], random.Random, int, JSONLFileState, BufShufBufferState, int, int, int]:
        """Set up worker-specific context."""
        worker = get_worker_info()
        global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)
        
        rng = random.Random()
        
        if resume_state is not None:
            if resume_state["global_worker_id"] != global_worker_id:
                raise ValueError(
                    f"Resume state worker ID mismatch: expected {global_worker_id}, "
                    f"got {resume_state['global_worker_id']}"
                )
            if resume_state["total_workers"] != total_workers:
                raise ValueError(
                    f"Resume state total workers mismatch: expected {total_workers}, "
                    f"got {resume_state['total_workers']}"
                )
            
            rng.setstate(resume_state["rng_state"])
            worker_files = resume_state["files_order"].copy()
            current_file_idx = resume_state["current_file_idx"]
            file_state = resume_state["file_state"].copy()
            buffer_state = resume_state["buffer_state"].copy()
            # Ensure token_buffer is a copy to avoid accidental shared refs
            buffer_state = BufShufBufferState(
                token_buffer=buffer_state.get("token_buffer", []).copy(),
                sample_buffer=buffer_state.get("sample_buffer", []).copy(),
                sample_buffer_idx=buffer_state.get("sample_buffer_idx", 0),
                documents_processed=buffer_state.get("documents_processed", 0),
            )
            epoch = resume_state["epoch"]
        else:
            if worker is not None:
                rng.seed(worker.id + 12345)
            else:
                rng.seed(os.getpid())
            
            # Prefer file-level sharding. Only use line-level sharding when
            # there are fewer files than workers.
            if len(files) >= total_workers:
                worker_files = files[global_worker_id::total_workers].copy()
                block_size = 1
                offset = 0
            else:
                worker_files = files.copy()
                block_size = total_workers
                offset = global_worker_id % total_workers

            if shuffle:
                rng.shuffle(worker_files)

            current_file_idx = 0

            file_state = JSONLFileState(
                file_path=worker_files[0] if worker_files else "",
                position=0,
                line_number=0,
                block_size=block_size,
                offset=offset,
                current_iter=0,
            )
            
            buffer_state = BufShufBufferState(
                token_buffer=[],
                sample_buffer=[],
                sample_buffer_idx=0,
                documents_processed=0,
            )
            
            epoch = 0

        self._current_state = BufShufSJSONLDatasetState(
            current_file_idx=current_file_idx,
            files_order=worker_files,
            file_state=file_state,
            buffer_state=buffer_state,
            rng_state=rng.getstate(),
            epoch=epoch,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
            tokenizer_config=self.tokenizer_config,
        )

        return (
            self._current_state["files_order"],
            rng,
            self._current_state["current_file_idx"],
            self._current_state["file_state"],
            self._current_state["buffer_state"],
            self._current_state["epoch"],
            self._current_state["global_worker_id"],
            self._current_state["total_workers"],
        )

    def _update_state(
        self,
        current_file_idx: int,
        files_order: List[str],
        file_state: JSONLFileState,
        buffer_state: BufShufBufferState,
        rng: random.Random,
        epoch: int,
        global_worker_id: int,
        total_workers: int,
    ) -> None:
        """Update internal state for checkpointing."""
        self._current_state = BufShufSJSONLDatasetState(
            current_file_idx=current_file_idx,
            files_order=files_order,
            file_state=file_state,
            buffer_state=buffer_state,
            rng_state=rng.getstate(),
            epoch=epoch,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
            tokenizer_config=self.tokenizer_config,
        )

    def _next_sample_from_file_state(
        self,
        line_iter: Iterator[tuple[dict, int, int]],
        file_state: JSONLFileState,
        token_buffer: List[int],
        documents_processed: int,
    ) -> tuple[Optional[dict], JSONLFileState, List[int], int]:
        """Return exactly one fixed-length sample, consuming tokens from token_buffer.

        This function appends tokens from `line_iter` until at least one sample
        can be produced. It returns the produced sample and the updated
        token_buffer (with consumed tokens removed).
        """
        while True:
            if len(token_buffer) >= self.seq_len + 1:
                sequence = token_buffer[: self.seq_len + 1]
                sample = dict(input_ids=sequence[:-1], labels=sequence[1:])
                # consume seq_len tokens (allow overlap of 1 token)
                token_buffer = token_buffer[self.seq_len :]
                return sample, file_state, token_buffer, documents_processed

            try:
                json_data, next_position, next_line_number = next(line_iter)
            except StopIteration:
                return None, file_state, token_buffer, documents_processed

            text_key = self.text_key if self.text_key in json_data else "content"
            if text_key not in json_data:
                continue

            text = json_data[text_key]
            tokens = self._tokenize_text(text)
            token_buffer.extend(tokens)
            documents_processed += 1

            file_state = JSONLFileState(
                file_path=file_state["file_path"],
                position=next_position,
                line_number=next_line_number,
                block_size=file_state["block_size"],
                offset=file_state["offset"],
                current_iter=file_state["current_iter"],
            )

    def _process_file_with_buffer(
        self,
        file_path: str,
        file_state: JSONLFileState,
        buffer_state: BufShufBufferState,
        worker_files: List[str],
        current_file_idx: int,
        rng: random.Random,
        epoch: int,
        global_worker_id: int,
        total_workers: int,
    ) -> Iterator[dict]:
        """
        Process one file using a rolling sample buffer.

        Behavior:
        1. Fill sample_buffer up to sample_buffer_size.
        2. Yield one sample at sample_buffer_idx.
        3. Replace emitted slot immediately with a new sample when available.
        4. When pointer wraps to the beginning, shuffle the whole buffer.
        """
        token_buffer = buffer_state["token_buffer"]
        sample_buffer = buffer_state["sample_buffer"]
        sample_buffer_idx = buffer_state["sample_buffer_idx"]
        documents_processed = buffer_state["documents_processed"]
        loaded_buffer = len(sample_buffer) > 0
        if sample_buffer_idx >= len(sample_buffer):
            sample_buffer_idx = 0

        line_iter = self._read_jsonl_lines(
            file_path=file_state["file_path"],
            position=file_state["position"],
            line_number=file_state["line_number"],
            block_size=file_state["block_size"],
            offset=file_state["offset"],
        )

        while len(sample_buffer) < self.sample_buffer_size:
            next_sample, file_state, token_buffer, documents_processed = (
                self._next_sample_from_file_state(
                    line_iter,
                    file_state,
                    token_buffer,
                    documents_processed,
                )
            )
            if next_sample is None:
                break
            sample_buffer.append(next_sample)

        if len(sample_buffer) == self.sample_buffer_size and not loaded_buffer:
            rng.shuffle(sample_buffer)

        while sample_buffer:
            if sample_buffer_idx >= len(sample_buffer):
                sample_buffer_idx = 0
                if len(sample_buffer) > 1:
                    rng.shuffle(sample_buffer)

            emit_idx = sample_buffer_idx
            emitted_sample = sample_buffer[emit_idx]

            replacement_sample, file_state, token_buffer, documents_processed = (
                self._next_sample_from_file_state(
                    line_iter,
                    file_state,
                    token_buffer,
                    documents_processed,
                )
            )
            if replacement_sample is not None:
                sample_buffer[emit_idx] = replacement_sample
                sample_buffer_idx += 1
            else:
                sample_buffer.pop(emit_idx)
                if sample_buffer_idx >= len(sample_buffer):
                    sample_buffer_idx = 0
            
            # No offset tracking: token_buffer is maintained as the authoritative
            # remaining tokens after consumption.
            self._current_state["buffer_state"]["token_buffer"] = token_buffer
            self._current_state["buffer_state"]["sample_buffer"] = sample_buffer
            self._current_state["buffer_state"]["sample_buffer_idx"] = sample_buffer_idx
            self._current_state["buffer_state"]["documents_processed"] = documents_processed
            

            yield emitted_sample
        
        # Update final state
        # Final buffer state: remaining token_buffer saved as-is.
        final_buffer_state = BufShufBufferState(
            token_buffer=token_buffer.copy(),
            sample_buffer=[],
            sample_buffer_idx=0,
            documents_processed=documents_processed,
        )
        
        self._update_state(
            current_file_idx=current_file_idx,
            files_order=worker_files,
            file_state=file_state,
            buffer_state=final_buffer_state,
            rng=rng,
            epoch=epoch,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
        )

    def _iterate_files(
        self,
        worker_files: List[str],
        rng: random.Random,
        current_file_idx: int,
        file_state: JSONLFileState,
        buffer_state: BufShufBufferState,
        epoch: int,
        global_worker_id: int,
        total_workers: int,
    ) -> Iterator[dict]:
        """Iterate through all files with shuffling."""
        while True:
            for file_idx in range(current_file_idx, len(worker_files)):
                file_path = worker_files[file_idx]
                
                if file_idx != current_file_idx or file_state["file_path"] != file_path:
                    file_state = JSONLFileState(
                        file_path=file_path,
                        position=0,
                        line_number=0,
                        block_size=file_state["block_size"],
                        offset=file_state["offset"],
                        current_iter=0,
                    )
                
                yield from self._process_file_with_buffer(
                    file_path=file_path,
                    file_state=file_state,
                    buffer_state=buffer_state,
                    worker_files=worker_files,
                    current_file_idx=file_idx,
                    rng=rng,
                    epoch=epoch,
                    global_worker_id=global_worker_id,
                    total_workers=total_workers,
                )
                
                buffer_state = self._current_state["buffer_state"]
            
            # Epoch complete
            epoch += 1
            current_file_idx = 0
            
            file_state = JSONLFileState(
                file_path=worker_files[0],
                position=0,
                line_number=0,
                block_size=file_state["block_size"],
                offset=file_state["offset"],
                current_iter=file_state["current_iter"] + 1,
            )

            if self.shuffle_files:
                rng.shuffle(worker_files)
                file_state = JSONLFileState(
                    file_path=worker_files[0],
                    position=0,
                    line_number=0,
                    block_size=file_state["block_size"],
                    offset=file_state["offset"],
                    current_iter=file_state["current_iter"],
                )

    def __iter__(self) -> Iterator[dict]:
        """Iterate over training samples with shuffling."""
        (
            worker_files,
            rng,
            current_file_idx,
            file_state,
            buffer_state,
            epoch,
            global_worker_id,
            total_workers,
        ) = self._setup_worker_context(self.files, self.shuffle_files, self.resume_state)

        # # Mock random sample for debugging:
        # while True:
        #     random_ids = [rng.randint(0, 100) for _ in range(self.seq_len)]
        #     yield dict(input_ids=random_ids[:-1], labels=random_ids[1:])
        
        yield from self._iterate_files(
            worker_files=worker_files,
            rng=rng,
            current_file_idx=current_file_idx,
            file_state=file_state,
            buffer_state=buffer_state,
            epoch=epoch,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
        )

    def __len__(self) -> int:
        raise NotImplementedError("__len__ is not implemented for BufShufSJSONLDataset.")

    def __getitem__(self, index: int):
        raise NotImplementedError("__getitem__ is not implemented for BufShufSJSONLDataset.")

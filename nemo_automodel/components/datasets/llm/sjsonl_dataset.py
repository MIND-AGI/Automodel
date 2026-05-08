"""
Stateful JSONL Dataset with file seeking and buffer management.

This dataset reads JSONL files line-by-line using file.seek() for efficient resumption,
similar to the lingua implementation. It supports:
- Multi-worker data loading with disjoint data assignment
- Token buffer management with state tracking
- Checkpoint/resume via state_dict/load_state_dict
- Integration with StatefulDataLoader
"""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path
from copy import deepcopy
from typing import Iterator, List, Sequence, Optional, TypedDict, Dict, Any

import torch
from torch.utils.data import IterableDataset, get_worker_info


__all__ = ["SJSONLDataset", "JSONLDatasetState"]


class JSONLFileState(TypedDict):
    """State for a single JSONL file being read.
    
    Attributes:
        file_path: Path to the JSONL file
        position: File position in bytes (for file.seek())
        line_number: Current line number being read
        block_size: Number of workers reading this file
        offset: Worker's offset for interleaved reading
        current_iter: Number of complete iterations through the file
    """
    file_path: str
    position: int
    line_number: int
    block_size: int
    offset: int
    current_iter: int


class BufferState(TypedDict):
    """State for the token buffer.
    
    Attributes:
        buffer: List of tokens currently in buffer
        buffer_token_offset: Offset within buffer where next sample starts
    """
    buffer: List[int]
    buffer_token_offset: int


class JSONLDatasetState(TypedDict):
    """Complete state for JSONL dataset checkpointing.
    
    Attributes:
        current_file_idx: Index of current file in files_order
        files_order: Ordered list of file paths (reflects shuffling)
        file_state: State of current JSONL file being read
        buffer_state: State of token buffer
        rng_state: Random number generator state
        epoch: Current epoch number
        global_worker_id: ID of this worker
        total_workers: Total number of workers
        tokenizer_config: Configuration for tokenizer
    """
    current_file_idx: int
    files_order: List[str]
    file_state: JSONLFileState
    buffer_state: BufferState
    rng_state: dict
    epoch: int
    global_worker_id: int
    total_workers: int
    tokenizer_config: Dict[str, Any]


def _get_worker_id_and_total_workers(worker: Optional[Any]) -> tuple[int, int]:
    """
    Get global worker ID and total workers across distributed training.
    
    Takes into account both DDP rank and DataLoader worker ID.
    """
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


class SJSONLDataset(IterableDataset):
    """
    Stateful JSONL Dataset with buffer management and file seeking.
    
    Reads JSONL files line-by-line, tokenizes on-the-fly, and manages a token buffer
    to produce fixed-length sequences. Supports checkpointing and resumption via
    state_dict/load_state_dict compatible with StatefulDataLoader.
    
    Args:
        file_pattern: Glob pattern or list of JSONL file paths
        seq_len: Length of sequences to produce
        tokenizer: Tokenizer instance with encode() method
        tokenizer_config: Dict with tokenizer configuration (name, path, add_bos, add_eos)
        shuffle_files: Whether to shuffle file order each epoch
        text_key: Key in JSONL for text content (default: "text", falls back to "content")
        resume_state: Optional state dict to resume from checkpoint
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
        resume_state: Optional[JSONLDatasetState] = None,
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
        
        # Internal state tracking
        self._current_state: Optional[JSONLDatasetState] = None

    def _normalize_token_buffer(self, token_buffer: List[int], token_buffer_offset: int) -> tuple[List[int], int]:
        """Compact the token buffer in place only when the skipped prefix is large."""
        if token_buffer_offset > len(token_buffer) // 2 and token_buffer_offset > 0:
            del token_buffer[:token_buffer_offset]
            return token_buffer, 0
        return token_buffer, token_buffer_offset

    def state_dict(self) -> JSONLDatasetState:
        """
        Returns current state for checkpointing.
        
        Called by StatefulDataLoader to capture worker-level state.
        Returns minimal valid state if iteration hasn't started yet.
        """
        if self._current_state is None:
            worker = get_worker_info()
            global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)
            if len(self.files) >= total_workers:
                block_size, offset = 1, 0
            else:
                block_size, offset = total_workers, global_worker_id % total_workers

            # Initialize _current_state for uninitialized workers
            self._current_state = JSONLDatasetState(
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
                buffer_state=BufferState(
                    buffer=[],
                    buffer_token_offset=0,
                ),
                rng_state=random.Random().getstate(),
                epoch=0,
                global_worker_id=global_worker_id,
                total_workers=total_workers,
                tokenizer_config=self.tokenizer_config,
            )

        return deepcopy(self._current_state)

    def load_state_dict(self, state: JSONLDatasetState) -> None:
        """
        Load a previously saved state to resume iteration.
        
        Args:
            state: JSONLDatasetState from a previous checkpoint.
        """
        # Snapshot the provided state as the authoritative current runtime state
        self._current_state = deepcopy(state)
        self.resume_state = None

    def _read_jsonl_lines(
        self,
        file_path: str,
        position: int,
        line_number: int,
        block_size: int,
        offset: int,
    ) -> Iterator[tuple[dict, int, int]]:
        """
        Read lines from JSONL file using file.seek() for resumption.
        
        Reads lines in an interleaved manner based on block_size and offset.
        Similar to lingua's read_jsonl implementation.
        
        Args:
            file_path: Path to JSONL file
            position: Byte position to seek to
            line_number: Current line number
            block_size: Number of workers reading this file
            offset: This worker's offset
            
        Yields:
            Tuple of (parsed_json, next_position, next_line_number)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            # Seek to saved position
            f.seek(position)
            
            # If starting from position > 0, we're resuming mid-file
            # line_number already accounts for lines we've processed
            current_line = line_number
            
            while True:
                line = f.readline()
                if not line:
                    # End of file
                    break
                
                # Check if this line is for this worker (interleaved reading)
                if current_line % block_size == offset:
                    # This line is for us
                    try:
                        data = json.loads(line)
                        next_position = f.tell()
                        next_line_number = current_line + 1
                        yield data, next_position, next_line_number
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        pass
                
                current_line += 1

    def _tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text using the configured tokenizer.
        
        Follows the official nanogpt_data_processor.py implementation:
        - Tokenize the text with the tokenizer
        - If BOS token is not already at the start and add_bos is True, prepend it
        - If EOS token should be added and add_eos is True, append it
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        # Get configuration
        add_bos = self.tokenizer_config.get("add_bos", False)
        add_eos = self.tokenizer_config.get("add_eos", False)
        max_length = self.tokenizer_config.get("max_length", None)
        
        # Tokenize (HuggingFace tokenizers don't have add_bos/add_eos parameters)
        tokens = self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=(max_length is not None),
        )
        
        # Manually add BOS token if needed (following official implementation)
        if add_bos and tokens and tokens[0] != self.tokenizer.bos_token_id:
            tokens = [self.tokenizer.bos_token_id] + tokens
        
        # Manually add EOS token if needed
        if add_eos and tokens and tokens[-1] != self.tokenizer.eos_token_id:
            tokens = tokens + [self.tokenizer.eos_token_id]
        
        return tokens

    def _setup_worker_context(
        self,
        files: List[str],
        shuffle: bool,
        resume_state: Optional[JSONLDatasetState],
    ) -> None:
        """Set up worker-specific context and initialize `_current_state`."""
        worker = get_worker_info()
        global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)

        rng = random.Random()

        if resume_state is not None:
            # Restore authoritative runtime state
            self._current_state = deepcopy(resume_state)
            state = self._current_state
            if state["global_worker_id"] != global_worker_id:
                raise ValueError(
                    f"Resume state worker ID mismatch: expected {global_worker_id}, "
                    f"got {state['global_worker_id']}"
                )
            if state["total_workers"] != total_workers:
                raise ValueError(
                    f"Resume state total workers mismatch: expected {total_workers}, "
                    f"got {state['total_workers']}"
                )

            rng.setstate(state["rng_state"])
            # Normalize token buffer on resume
            buffer_state = state["buffer_state"]
            normalized_buffer, normalized_offset = self._normalize_token_buffer(
                buffer_state["buffer"], buffer_state["buffer_token_offset"]
            )
            buffer_state["buffer"] = normalized_buffer
            buffer_state["buffer_token_offset"] = normalized_offset
            state["buffer_state"] = buffer_state
        else:
            if worker is not None:
                rng.seed(worker.id + 12345)
            else:
                rng.seed(os.getpid())

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

            buffer_state = BufferState(
                buffer=[],
                buffer_token_offset=0,
            )

            epoch = 0
            self._current_state = JSONLDatasetState(
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

        # Ensure top-level fields reflect current worker context
        state = self._current_state
        state["global_worker_id"] = global_worker_id
        state["total_workers"] = total_workers
        state["rng_state"] = rng.getstate()

    def _update_state(
        self,
        current_file_idx: int,
        files_order: List[str],
        file_state: JSONLFileState,
        buffer_state: BufferState,
        rng: random.Random,
        epoch: int,
        global_worker_id: int,
        total_workers: int,
    ) -> None:
        """Update internal state for checkpointing."""
        # Keep this helper for external explicit snapshots, but avoid hot-path copies.
        self._current_state = JSONLDatasetState(
            current_file_idx=current_file_idx,
            files_order=files_order.copy(),
            file_state=deepcopy(file_state),
            buffer_state=deepcopy(buffer_state),
            rng_state=rng.getstate(),
            epoch=epoch,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
            tokenizer_config=self.tokenizer_config,
        )

    def _process_file(
        self,
    ) -> Iterator[dict]:
        """
        Process a single file, managing buffer and yielding samples.
        
        This implements the core buffer management logic similar to lingua's
        pack_tokens function, but integrated with file reading.
        
        Args:
            file_path: Path to current JSONL file
            file_state: Current file reading state
            buffer_state: Current buffer state
            (other args for state tracking)
            
        Yields:
            Training samples (dicts with 'input_ids' and 'labels')
        """
        state = self._current_state
        file_state = state["file_state"]
        buffer_state = state["buffer_state"]
        rng = random.Random()
        rng.setstate(state["rng_state"])
        current_file_idx = state["current_file_idx"]
        worker_files = state["files_order"]
        epoch = state["epoch"]
        global_worker_id = state["global_worker_id"]
        total_workers = state["total_workers"]

        token_buffer = buffer_state["buffer"]
        buffer_offset = buffer_state["buffer_token_offset"]
        
        # Read lines from file
        for json_data, next_position, next_line_number in self._read_jsonl_lines(
            file_path=file_state["file_path"],
            position=file_state["position"],
            line_number=file_state["line_number"],
            block_size=file_state["block_size"],
            offset=file_state["offset"],
        ):
            # Extract and tokenize text
            text_key = self.text_key if self.text_key in json_data else "content"
            if text_key not in json_data:
                continue
            
            text = json_data[text_key]
            tokens = self._tokenize_text(text)
            
            # Add tokens to buffer
            token_buffer.extend(tokens)
            
            # Update file state for next iteration
            file_state = JSONLFileState(
                file_path=file_state["file_path"],
                position=next_position,
                line_number=next_line_number,
                block_size=file_state["block_size"],
                offset=file_state["offset"],
                current_iter=file_state["current_iter"],
            )
            state["file_state"] = file_state
            
            # Yield samples from buffer while we have enough tokens
            while len(token_buffer) - buffer_offset >= self.seq_len + 1:
                # Extract sequence
                start_idx = buffer_offset
                end_idx = start_idx + self.seq_len + 1

                sequence = token_buffer[start_idx:end_idx]
                inputs = sequence[:-1]
                labels = sequence[1:]
                
                # 1-token overlap: stride by seq_len
                buffer_offset += self.seq_len
                # Update runtime buffer offsets in-place (avoid copies)
                buffer_state["buffer"] = token_buffer
                buffer_state["buffer_token_offset"] = buffer_offset
                state["buffer_state"] = buffer_state
                state["file_state"] = file_state
                yield dict(input_ids=inputs, labels=labels)
            
            # Compact buffer in-place if offset is large (keep only remaining tokens)
            if buffer_offset > len(token_buffer) // 2 and buffer_offset > 0:
                del token_buffer[:buffer_offset]
                buffer_offset = 0
                buffer_state["buffer"] = token_buffer
                buffer_state["buffer_token_offset"] = buffer_offset
                state["buffer_state"] = buffer_state
        
        # File exhausted - update buffer state for next file
        # Keep remaining tokens in buffer for next file
        # Update final state with normalized buffer
        normalized_buffer, normalized_offset = self._normalize_token_buffer(token_buffer, buffer_offset)
        buffer_state["buffer"] = normalized_buffer
        buffer_state["buffer_token_offset"] = normalized_offset
        state["current_file_idx"] = current_file_idx
        state["files_order"] = worker_files
        state["file_state"] = file_state
        state["buffer_state"] = buffer_state
        state["rng_state"] = rng.getstate()
        state["epoch"] = epoch
        state["global_worker_id"] = global_worker_id
        state["total_workers"] = total_workers

    def _iterate_files(
        self,
    ) -> Iterator[dict]:
        """
        Iterate through all files, handling epoch boundaries and shuffling.
        
        Yields:
            Training samples from all files
        """
        while True:
            # Process files starting from current_file_idx
            state = self._current_state
            worker_files = state["files_order"]
            current_file_idx = state["current_file_idx"]
            file_state = state["file_state"]
            for file_idx in range(current_file_idx, len(worker_files)):
                file_path = worker_files[file_idx]

                # Update file state if moving to new file
                if file_idx != current_file_idx or file_state["file_path"] != file_path:
                    file_state = JSONLFileState(
                        file_path=file_path,
                        position=0,
                        line_number=0,
                        block_size=file_state["block_size"],
                        offset=file_state["offset"],
                        current_iter=0,
                    )
                    state["file_state"] = file_state
                    state["current_file_idx"] = file_idx

                # Process this file
                yield from self._process_file()
                state = self._current_state
            
            # Epoch complete - reset for next epoch
            state = self._current_state
            state["epoch"] = state.get("epoch", 0) + 1
            state["current_file_idx"] = 0

            file_state = JSONLFileState(
                file_path=state["files_order"][0],
                position=0,
                line_number=0,
                block_size=state["file_state"]["block_size"],
                offset=state["file_state"]["offset"],
                current_iter=state["file_state"]["current_iter"] + 1,
            )
            state["file_state"] = file_state

            # Optionally reshuffle files
            if self.shuffle_files:
                rng = random.Random()
                rng.setstate(state["rng_state"])
                rng.shuffle(state["files_order"])
                state["rng_state"] = rng.getstate()
                file_state = JSONLFileState(
                    file_path=state["files_order"][0],
                    position=0,
                    line_number=0,
                    block_size=state["file_state"]["block_size"],
                    offset=state["file_state"]["offset"],
                    current_iter=state["file_state"]["current_iter"],
                )
                state["file_state"] = file_state

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over training samples.
        
        Yields:
            Dictionary containing 'input_ids' and 'labels'
        """
        # Initialize or restore worker runtime state
        self._setup_worker_context(self.files, self.shuffle_files, self.resume_state)
        yield from self._iterate_files()

    def __len__(self) -> int:
        raise NotImplementedError("__len__ is not implemented for SJSONLDataset.")

    def __getitem__(self, index: int):
        raise NotImplementedError("__getitem__ is not implemented for SJSONLDataset.")
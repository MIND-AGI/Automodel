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

    def state_dict(self) -> JSONLDatasetState:
        """
        Returns current state for checkpointing.
        
        Called by StatefulDataLoader to capture worker-level state.
        Returns minimal valid state if iteration hasn't started yet.
        """
        if self._current_state is None:
            # Return minimal valid state for uninitialized workers
            worker = get_worker_info()
            global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)
            if len(self.files) >= total_workers:
                block_size, offset = 1, 0
            else:
                block_size, offset = total_workers, global_worker_id % total_workers
            
            return JSONLDatasetState(
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
        
        return self._current_state.copy()

    def load_state_dict(self, state: JSONLDatasetState) -> None:
        """
        Load a previously saved state to resume iteration.
        
        Args:
            state: JSONLDatasetState from a previous checkpoint.
        """
        self.resume_state = state

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
    ) -> tuple[List[str], random.Random, int, JSONLFileState, BufferState, int, int, int]:
        """
        Set up worker-specific context.
        
        Returns:
            Tuple of (worker_files, rng, current_file_idx, file_state, 
                     buffer_state, epoch, global_worker_id, total_workers)
        """
        worker = get_worker_info()
        global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)
        
        rng = random.Random()
        
        if resume_state is not None:
            # Validate worker consistency
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
            
            # Restore state
            rng.setstate(resume_state["rng_state"])
            worker_files = resume_state["files_order"].copy()
            current_file_idx = resume_state["current_file_idx"]
            file_state = resume_state["file_state"].copy()
            buffer_state = resume_state["buffer_state"].copy()
            epoch = resume_state["epoch"]
        else:
            # Fresh start - deterministic seeding
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
            
            # Initialize file state for first file
            file_state = JSONLFileState(
                file_path=worker_files[0] if worker_files else "",
                position=0,
                line_number=0,
                block_size=block_size,
                offset=offset,
                current_iter=0,
            )
            
            # Initialize empty buffer
            buffer_state = BufferState(
                buffer=[],
                buffer_token_offset=0,
            )
            
            epoch = 0
        
        return (
            worker_files,
            rng,
            current_file_idx,
            file_state,
            buffer_state,
            epoch,
            global_worker_id,
            total_workers,
        )

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
        self._current_state = JSONLDatasetState(
            current_file_idx=current_file_idx,
            files_order=files_order.copy(),
            file_state=file_state.copy(),
            buffer_state=buffer_state.copy(),
            rng_state=rng.getstate(),
            epoch=epoch,
            global_worker_id=global_worker_id,
            total_workers=total_workers,
            tokenizer_config=self.tokenizer_config,
        )

    def _process_file_with_buffer(
        self,
        file_path: str,
        file_state: JSONLFileState,
        buffer_state: BufferState,
        worker_files: List[str],
        current_file_idx: int,
        rng: random.Random,
        epoch: int,
        global_worker_id: int,
        total_workers: int,
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
        # Extract buffer state
        buffer = buffer_state["buffer"].copy()
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
            buffer.extend(tokens)
            
            # Update file state for next iteration
            file_state = JSONLFileState(
                file_path=file_state["file_path"],
                position=next_position,
                line_number=next_line_number,
                block_size=file_state["block_size"],
                offset=file_state["offset"],
                current_iter=file_state["current_iter"],
            )
            
            # Yield samples from buffer while we have enough tokens
            while len(buffer) - buffer_offset >= self.seq_len + 1:
                # Extract sequence
                start_idx = buffer_offset
                end_idx = start_idx + self.seq_len + 1
                
                sequence = buffer[start_idx:end_idx]
                inputs = sequence[:-1]
                labels = sequence[1:]
                
                # Megatron-compatible 1-token overlap: stride by seq_len,
                # while each sample still uses seq_len + 1 tokens.
                buffer_offset += self.seq_len
                
                
                # Update state before yielding
                new_buffer_state = BufferState(
                    buffer=buffer.copy(),
                    buffer_token_offset=buffer_offset,
                )
                
                self._update_state(
                    current_file_idx=current_file_idx,
                    files_order=worker_files,
                    file_state=file_state,
                    buffer_state=new_buffer_state,
                    rng=rng,
                    epoch=epoch,
                    global_worker_id=global_worker_id,
                    total_workers=total_workers,
                )
                
                yield dict(input_ids=inputs, labels=labels)
            
            # Compact buffer if offset is large (keep only remaining tokens)
            if buffer_offset > len(buffer) // 2 and buffer_offset > 0:
                buffer = buffer[buffer_offset:]
                buffer_offset = 0
        
        # File exhausted - update buffer state for next file
        # Keep remaining tokens in buffer for next file
        final_buffer_state = BufferState(
            buffer=buffer.copy(),
            buffer_token_offset=buffer_offset,
        )
        
        # Update state with exhausted file (position at end, ready for next file)
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
        buffer_state: BufferState,
        epoch: int,
        global_worker_id: int,
        total_workers: int,
    ) -> Iterator[dict]:
        """
        Iterate through all files, handling epoch boundaries and shuffling.
        
        Yields:
            Training samples from all files
        """
        while True:
            # Process files starting from current_file_idx
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
                
                # Process this file
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
                
                # After processing file, reset file state for next file
                # Buffer state is preserved from _process_file_with_buffer
                buffer_state = self._current_state["buffer_state"].copy()
            
            # Epoch complete - reset for next epoch
            epoch += 1
            current_file_idx = 0
            
            # Reset file state for first file of new epoch
            file_state = JSONLFileState(
                file_path=worker_files[0],
                position=0,
                line_number=0,
                block_size=file_state["block_size"],
                offset=file_state["offset"],
                current_iter=file_state["current_iter"] + 1,
            )
            
            # Optionally reshuffle files
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
        """
        Iterate over training samples.
        
        Yields:
            Dictionary containing 'input_ids' and 'labels'
        """
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
        raise NotImplementedError("__len__ is not implemented for SJSONLDataset.")

    def __getitem__(self, index: int):
        raise NotImplementedError("__getitem__ is not implemented for SJSONLDataset.")
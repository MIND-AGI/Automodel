# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_automodel.components.datasets.llm.blended_jsonl_dataset import BlendedJSONLDataset


class DummyTokenizer:
    bos_token_id = 101
    eos_token_id = 102

    def encode(self, text, max_length=None, truncation=False):
        tokens = [ord(char) % 97 + 1 for char in text]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return tokens


def _sample_signature(sample: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(sample["input_ids"]), tuple(sample["labels"])


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _write_mock_sources(tmp_path: Path, *, files_per_source: int, docs_per_file: int) -> tuple[str, str]:
    source_a_dir = tmp_path / "source_a"
    source_b_dir = tmp_path / "source_b"
    source_a_dir.mkdir(parents=True, exist_ok=True)
    source_b_dir.mkdir(parents=True, exist_ok=True)

    for source_name, source_dir in (("source_a", source_a_dir), ("source_b", source_b_dir)):
        for file_idx in range(files_per_source):
            path = source_dir / f"part_{file_idx}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for doc_idx in range(docs_per_file):
                    if source_name == "source_a":
                        text = f"A-{file_idx}-{doc_idx}-" + ("abcdefghij" * 14)
                    else:
                        text = f"B-{file_idx}-{doc_idx}-" + ("uvwxyz0123" * 14)
                    f.write(json.dumps({"text": text}) + "\n")

    return str(source_a_dir / "*.jsonl"), str(source_b_dir / "*.jsonl")


def _run_blended_worker(
    rank: int,
    world_size: int,
    port: int,
    dataset_type: str,
    source_a_pattern: str,
    source_b_pattern: str,
    return_dict,
):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )

    try:
        dataset = BlendedJSONLDataset(
            sources={source_a_pattern: 50.0, source_b_pattern: 50.0},
            seq_len=16,
            tokenizer=DummyTokenizer(),
            tokenizer_config={"add_bos": True, "add_eos": True, "max_length": 256},
            dataset_type=dataset_type,
            shuffle_files=False,
            sample_buffer_size=8,
            prefetch_factor=2,
            random_seed=2026,
        )

        dataloader = StatefulDataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        iterator = iter(dataloader)
        advance_count = 24
        compare_count = 32

        for _ in range(advance_count):
            next(iterator)

        checkpoint = dataloader.state_dict()
        continued = [_sample_signature(next(iterator)) for _ in range(compare_count)]

        resumed_dataset = BlendedJSONLDataset(
            sources={source_a_pattern: 50.0, source_b_pattern: 50.0},
            seq_len=16,
            tokenizer=DummyTokenizer(),
            tokenizer_config={"add_bos": True, "add_eos": True, "max_length": 256},
            dataset_type=dataset_type,
            shuffle_files=False,
            sample_buffer_size=8,
            prefetch_factor=2,
            random_seed=2026,
        )
        resumed_dataloader = StatefulDataLoader(
            resumed_dataset,
            batch_size=None,
            num_workers=0,
        )
        resumed_dataloader.load_state_dict(checkpoint)

        resumed_iter = iter(resumed_dataloader)
        resumed = [_sample_signature(next(resumed_iter)) for _ in range(compare_count)]

        return_dict[rank] = {
            "resume_matches": resumed == continued,
            "continued_head": continued[:10],
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dataset_type", ["sjsonl", "bufshuf_sjsonl"])
def test_blended_jsonl_stateful_dataloader_multirank_resume(tmp_path, dataset_type):
    """Integration test for real multi-rank startup with blended JSONL datasets.

    This test validates:
    1. Multi-rank process launch with torch.distributed.
    2. Real StatefulDataLoader state_dict/load_state_dict resume chain.
    3. Resume continuity for both blended+sjsonl and blended+bufshuf_sjsonl.
    """
    source_a_pattern, source_b_pattern = _write_mock_sources(
        tmp_path,
        files_per_source=4,
        docs_per_file=40,
    )

    world_size = 2
    port = _find_free_port()
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        _run_blended_worker,
        args=(world_size, port, dataset_type, source_a_pattern, source_b_pattern, return_dict),
        nprocs=world_size,
        join=True,
    )

    rank0 = return_dict[0]
    rank1 = return_dict[1]

    assert rank0["resume_matches"]
    assert rank1["resume_matches"]

    # For file-sharded reading, two ranks should not produce identical streams.
    assert rank0["continued_head"] != rank1["continued_head"]
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

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

import nemo_automodel.components.datasets.llm.bufshuf_sjsonl_dataset as bufshuf_module
import nemo_automodel.components.datasets.llm.sjsonl_dataset as sjsonl_module
from nemo_automodel.components.datasets.llm.bufshuf_sjsonl_dataset import BufShufSJSONLDataset
from nemo_automodel.components.datasets.llm.sjsonl_dataset import SJSONLDataset


class DummyTokenizer:
    bos_token_id = 101
    eos_token_id = 102

    def encode(self, text, max_length=None, truncation=False):
        tokens = [ord(char) % 89 + 1 for char in text]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return tokens


DATASET_CASES = [
    (
        "sjsonl",
        SJSONLDataset,
        sjsonl_module,
        {},
    ),
    (
        "bufshuf_sjsonl",
        BufShufSJSONLDataset,
        bufshuf_module,
        {"sample_buffer_size": 5, "prefetch_factor": 2},
    ),
]


def _write_jsonl_files(tmp_path: Path, *, num_files: int, docs_per_file: int) -> list[str]:
    file_paths = []
    for file_idx in range(num_files):
        file_path = tmp_path / f"source_{file_idx}.jsonl"
        with file_path.open("w", encoding="utf-8") as f:
            for doc_idx in range(docs_per_file):
                payload = {
                    "text": f"file-{file_idx}-doc-{doc_idx}-" + chr(97 + file_idx) * 48,
                }
                f.write(json.dumps(payload) + "\n")
        file_paths.append(str(file_path))
    return file_paths


def _patch_execution_context(
    monkeypatch,
    module,
    *,
    rank: int,
    world_size: int,
    worker_id: int | None,
    num_workers: int = 1,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: world_size > 1, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank, raising=False)

    if worker_id is None:
        monkeypatch.setattr(module, "get_worker_info", lambda: None)
        return

    worker = SimpleNamespace(id=worker_id, num_workers=num_workers)
    monkeypatch.setattr(module, "get_worker_info", lambda: worker)


def _make_dataset(dataset_cls, file_paths: list[str], resume_state=None, extra_kwargs=None):
    kwargs = {
        "file_pattern": file_paths,
        "seq_len": 8,
        "tokenizer": DummyTokenizer(),
        "tokenizer_config": {"add_bos": True, "add_eos": True, "max_length": 128},
        "shuffle_files": False,
        "text_key": "text",
        "resume_state": resume_state,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return dataset_cls(**kwargs)


def _sample_signature(sample: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(sample["input_ids"]), tuple(sample["labels"])


def _capture_resume_pair(dataset, *, advance_count: int, compare_count: int):
    iterator = iter(dataset)
    for _ in range(advance_count):
        next(iterator)

    resume_state = dataset.state_dict()
    continued = [_sample_signature(next(iterator)) for _ in range(compare_count)]
    return resume_state, continued


@pytest.mark.parametrize("_name,dataset_cls,module,extra_kwargs", DATASET_CASES)
def test_streaming_jsonl_resume_matches_continuous_single_process(
    tmp_path,
    monkeypatch,
    _name,
    dataset_cls,
    module,
    extra_kwargs,
):
    _patch_execution_context(monkeypatch, module, rank=0, world_size=1, worker_id=None)
    file_paths = _write_jsonl_files(tmp_path, num_files=2, docs_per_file=30)

    dataset = _make_dataset(dataset_cls, file_paths, extra_kwargs=extra_kwargs)
    resume_state, continued = _capture_resume_pair(dataset, advance_count=25, compare_count=30)

    resumed_dataset = _make_dataset(dataset_cls, file_paths, resume_state=resume_state, extra_kwargs=extra_kwargs)
    resumed_iterator = iter(resumed_dataset)
    resumed = [_sample_signature(next(resumed_iterator)) for _ in range(30)]

    assert resumed == continued


@pytest.mark.parametrize("_name,dataset_cls,module,extra_kwargs", DATASET_CASES)
def test_streaming_jsonl_resume_matches_continuous_simulated_torchrun_rank(
    tmp_path,
    monkeypatch,
    _name,
    dataset_cls,
    module,
    extra_kwargs,
):
    _patch_execution_context(monkeypatch, module, rank=1, world_size=2, worker_id=None)
    file_paths = _write_jsonl_files(tmp_path, num_files=4, docs_per_file=24)

    dataset = _make_dataset(dataset_cls, file_paths, extra_kwargs=extra_kwargs)
    resume_state, continued = _capture_resume_pair(dataset, advance_count=18, compare_count=24)

    assert resume_state["global_worker_id"] == 1
    assert resume_state["total_workers"] == 2
    assert resume_state["files_order"] == file_paths[1::2]

    resumed_dataset = _make_dataset(dataset_cls, file_paths, resume_state=resume_state, extra_kwargs=extra_kwargs)
    resumed_iterator = iter(resumed_dataset)
    resumed = [_sample_signature(next(resumed_iterator)) for _ in range(24)]

    assert resumed == continued


@pytest.mark.parametrize("_name,dataset_cls,module,extra_kwargs", DATASET_CASES)
def test_streaming_jsonl_resume_matches_continuous_line_sharded_workers(
    tmp_path,
    monkeypatch,
    _name,
    dataset_cls,
    module,
    extra_kwargs,
):
    _patch_execution_context(monkeypatch, module, rank=1, world_size=2, worker_id=1, num_workers=2)
    file_paths = _write_jsonl_files(tmp_path, num_files=1, docs_per_file=80)

    dataset = _make_dataset(dataset_cls, file_paths, extra_kwargs=extra_kwargs)
    resume_state, continued = _capture_resume_pair(dataset, advance_count=12, compare_count=20)

    assert resume_state["global_worker_id"] == 3
    assert resume_state["total_workers"] == 4
    assert resume_state["file_state"]["block_size"] == 4
    assert resume_state["file_state"]["offset"] == 3

    resumed_dataset = _make_dataset(dataset_cls, file_paths, resume_state=resume_state, extra_kwargs=extra_kwargs)
    resumed_iterator = iter(resumed_dataset)
    resumed = [_sample_signature(next(resumed_iterator)) for _ in range(20)]

    assert resumed == continued
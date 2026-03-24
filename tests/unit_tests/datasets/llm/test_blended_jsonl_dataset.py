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

import json
from pathlib import Path

from nemo_automodel.components.datasets.llm.blended_jsonl_dataset import BlendedJSONLDataset
from nemo_automodel.components.datasets.llm.bufshuf_sjsonl_dataset import BufShufSJSONLDataset
from nemo_automodel.components.datasets.llm.sjsonl_dataset import SJSONLDataset


class DummyTokenizer:
    bos_token_id = 101
    eos_token_id = 102

    def encode(self, text, max_length=None, truncation=False):
        tokens = [ord(char) % 97 + 1 for char in text]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return tokens


def _write_jsonl(path: Path, prefix: str, count: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx in range(count):
            row = {"text": f"{prefix}-doc-{idx}-" + prefix * 24}
            f.write(json.dumps(row) + "\n")


def _make_dataset(tmp_path: Path, dataset_type: str, resume_state=None) -> BlendedJSONLDataset:
    source_a = tmp_path / "source_a.jsonl"
    source_b = tmp_path / "source_b.jsonl"
    _write_jsonl(source_a, "a", 20)
    _write_jsonl(source_b, "b", 20)

    return BlendedJSONLDataset(
        sources=["30", str(source_a), "70", str(source_b)],
        seq_len=8,
        tokenizer=DummyTokenizer(),
        tokenizer_config={"add_bos": True, "add_eos": True, "max_length": 128},
        dataset_type=dataset_type,
        shuffle_files=False,
        sample_buffer_size=5,
        random_seed=12345,
        resume_state=resume_state,
    )


def _materialize_samples(dataset: BlendedJSONLDataset, count: int):
    iterator = iter(dataset)
    samples = []
    for _ in range(count):
        sample = next(iterator)
        samples.append((tuple(sample["input_ids"]), tuple(sample["labels"])))
    return samples


def test_blended_jsonl_dataset_respects_dataset_type(tmp_path):
    sjsonl_dataset = _make_dataset(tmp_path, dataset_type="sjsonl")
    bufshuf_dataset = _make_dataset(tmp_path, dataset_type="bufshuf_sjsonl")

    assert isinstance(sjsonl_dataset._build_source_dataset(str(tmp_path / "source_a.jsonl"), None), SJSONLDataset)
    assert isinstance(
        bufshuf_dataset._build_source_dataset(str(tmp_path / "source_a.jsonl"), None),
        BufShufSJSONLDataset,
    )


def test_blended_jsonl_dataset_resume_matches_continuous_stream_sjsonl(tmp_path):
    dataset = _make_dataset(tmp_path, dataset_type="sjsonl")
    iterator = iter(dataset)

    for _ in range(25):
        next(iterator)

    resume_state = dataset.state_dict()
    continued_samples = [
        (tuple(sample["input_ids"]), tuple(sample["labels"])) for sample in (next(iterator) for _ in range(30))
    ]

    resumed_dataset = _make_dataset(tmp_path, dataset_type="sjsonl", resume_state=resume_state)
    resumed_samples = _materialize_samples(resumed_dataset, 30)

    assert resumed_samples == continued_samples


def test_blended_jsonl_dataset_resume_matches_continuous_stream_bufshuf(tmp_path):
    dataset = _make_dataset(tmp_path, dataset_type="bufshuf_sjsonl")
    iterator = iter(dataset)

    for _ in range(25):
        next(iterator)

    resume_state = dataset.state_dict()
    continued_samples = [
        (tuple(sample["input_ids"]), tuple(sample["labels"])) for sample in (next(iterator) for _ in range(30))
    ]

    resumed_dataset = _make_dataset(tmp_path, dataset_type="bufshuf_sjsonl", resume_state=resume_state)
    resumed_samples = _materialize_samples(resumed_dataset, 30)

    assert resumed_samples == continued_samples
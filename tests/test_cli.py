from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import EngramCompiler
from engram_rs.cli import build_parser, main, read_input_text


class DummyByteTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        token_ids = [1, *(byte % 200 + 2 for byte in text.encode("utf-8"))]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def encode(self, text: str) -> list[int]:
        return [1, *(byte % 200 + 2 for byte in text.encode("utf-8"))]

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "Decoded needle response"


def create_saved_tiny_llama(path: Path) -> Path:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        _name_or_path=str(path),
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(str(path))
    return path


def test_read_input_text_from_file_and_dir(tmp_path: Path):
    # Single file
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Hello from document file.", encoding="utf-8")
    assert read_input_text(str(file_path)) == "Hello from document file."

    # Directory with multiple files
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "a.md").write_text("Markdown section A", encoding="utf-8")
    (doc_dir / "b.txt").write_text("Text section B", encoding="utf-8")
    combined = read_input_text(str(doc_dir))
    assert "Markdown section A" in combined
    assert "Text section B" in combined

    with pytest.raises(FileNotFoundError):
        read_input_text(str(tmp_path / "nonexistent"))


def test_cli_compile_and_export_gguf_flow(tmp_path: Path):
    model_dir = tmp_path / "base_model"
    create_saved_tiny_llama(model_dir)

    input_file = tmp_path / "input.txt"
    input_file.write_text("Secret lab located in the mountains.", encoding="utf-8")
    adapter_dir = tmp_path / "out_adapter"

    # Patch AutoTokenizer so it doesn't need network
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyByteTokenizer()):
        code = main([
            "compile",
            "--model", str(model_dir),
            "--input", str(input_file),
            "--output", str(adapter_dir),
            "--rank", "2",
            "--device", "cpu",
        ])
        assert code == 0

    assert (adapter_dir / "adapter_model.safetensors").is_file()
    assert (adapter_dir / "adapter_config.json").is_file()

    # Now test export-gguf
    gguf_file = tmp_path / "adapter.gguf"
    code = main([
        "export-gguf",
        "--adapter", str(adapter_dir),
        "--output", str(gguf_file),
        "--out-type", "f32",
    ])
    assert code == 0
    assert gguf_file.is_file()


def test_cli_merge_flow(tmp_path: Path):
    model_dir = tmp_path / "base_model"
    create_saved_tiny_llama(model_dir)

    input_file = tmp_path / "context.txt"
    input_file.write_text("Merging test activations.", encoding="utf-8")
    adapter_dir = tmp_path / "adapter"

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyByteTokenizer()):
        code_compile = main([
            "compile",
            "--model", str(model_dir),
            "--input", str(input_file),
            "--output", str(adapter_dir),
            "--rank", "2",
            "--device", "cpu",
        ])
        assert code_compile == 0

    merged_dir = tmp_path / "merged_model"
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyByteTokenizer()):
        code_merge = main([
            "merge",
            "--base", str(model_dir),
            "--adapter", str(adapter_dir),
            "--output", str(merged_dir),
            "--device", "cpu",
        ])
        assert code_merge == 0

    assert (merged_dir / "model.safetensors").is_file()
    assert (merged_dir / "config.json").is_file()


def test_cli_compare_rag_flow(tmp_path: Path, capsys):
    out_json = tmp_path / "comparison.json"
    code = main([
        "compare-rag",
        "--context-len", "10000",
        "--queries", "1,10,100",
        "-o", str(out_json),
    ])
    assert code == 0
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "parameters" in data
    assert "query_cost_curves" in data
    assert len(data["query_cost_curves"]) == 3
    assert data["parameters"]["context_tokens"] == 10000

    captured = capsys.readouterr()
    assert "ENGRAM-RS VS TRADITIONAL RAG" in captured.out


def test_cli_benchmark_niah_flow(tmp_path: Path, capsys):
    model_dir = tmp_path / "base_model"
    create_saved_tiny_llama(model_dir)

    out_json = tmp_path / "niah_results.json"
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyByteTokenizer()):
        code = main([
            "benchmark-niah",
            "-m", str(model_dir),
            "--context-lens", "50,100",
            "--depths", "0,50,100",
            "-r", "2",
            "-o", str(out_json),
        ])
        assert code == 0

    assert out_json.is_file()
    results = json.loads(out_json.read_text(encoding="utf-8"))
    assert results["benchmark"] == "Needle In A Haystack (NIAH)"
    assert "results" in results

    captured = capsys.readouterr()
    assert "NEEDLE IN A HAYSTACK BENCHMARK REPORT" in captured.out

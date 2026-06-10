"""Lightweight tests for the training entrypoint; uses monkeypatching to avoid running full training."""

import os
import sys

# Ensure local src/ is importable
TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(TEST_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from projects_30_ai_language_translation import train as train_module  # type: ignore


def test_preprocess_function():
    # Create a fake tokenizer with minimal interface used by preprocess_function
    class FakeTokenizer:
        def __init__(self):
            self.pad_token_id = 0

        def __call__(self, texts, max_length, truncation, padding):
            # return dummy input ids
            return {"input_ids": [[0] * max_length for _ in texts]}

        def as_target_tokenizer(self):
            # context manager stub
            class CTX:
                def __enter__(self_inner):
                    return self

                def __exit__(self_inner, exc_type, exc, tb):
                    return False
            return CTX()

    examples = {"src": ["hello"], "tgt": ["hola"]}
    tokenizer = FakeTokenizer()
    out = train_module.preprocess_function(examples, tokenizer, max_source_length=10, max_target_length=10)
    assert "labels" in out
    assert isinstance(out["labels"], list) or isinstance(out["labels"], list)

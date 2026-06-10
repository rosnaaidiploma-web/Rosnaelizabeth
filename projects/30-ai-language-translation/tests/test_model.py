"""Unit tests for the TranslationModel wrapper using mocking to avoid heavy downloads."""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Ensure the local src/ directory is importable as a package during tests
TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(TEST_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from projects_30_ai_language_translation.model import TranslationModel


@pytest.fixture(autouse=True)
def auto_patch_imports(monkeypatch):
    # Patch the transformers AutoTokenizer and AutoModelForSeq2SeqLM to avoid network calls.
    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = fake_tokenizer
    fake_tokenizer.__call__.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    fake_tokenizer.decode = lambda ids, skip_special_tokens, clean_up_tokenization_spaces: "translated sentence"

    fake_model = MagicMock()
    fake_model.generate.return_value = [[1, 2, 3]]

    monkeypatch.setattr("projects_30_ai_language_translation.model.AutoTokenizer.from_pretrained", lambda name: fake_tokenizer)
    monkeypatch.setattr("projects_30_ai_language_translation.model.AutoModelForSeq2SeqLM.from_pretrained", lambda name: fake_model)

    yield


def test_load_and_predict():
    tm = TranslationModel(model_name="fake-model")
    tm.load()
    out = tm.predict(["hello world"])
    assert isinstance(out, list)
    assert out[0] == "translated sentence"

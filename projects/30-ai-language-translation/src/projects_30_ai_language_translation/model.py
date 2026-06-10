"""Minimal model wrapper for seq2seq translation using Hugging Face Transformers."""

from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TranslationModel:
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        """Load tokenizer and model from pretrained checkpoint."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], src_lang: Optional[str] = None, tgt_lang: Optional[str] = None, max_length: int = 128, num_beams: int = 4) -> List[str]:
        """Translate a list of texts and return decoded outputs."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        # Optionally set language tokens if tokenizer supports it (e.g., MBART)
        # Users can set tokenizer.src_lang or tokenizer.tgt_lang beforehand for specific tokenizers.
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        decoded = [self.tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in outputs]
        return decoded

    def save(self, path: str):
        """Save tokenizer and model to directory."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Nothing to save. Load or train a model first.")
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

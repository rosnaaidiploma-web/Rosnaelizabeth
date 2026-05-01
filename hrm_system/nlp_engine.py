from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeAnalyzer:
    def __init__(self, known_skills: Iterable[str]) -> None:
        self.known_skills = sorted({skill.strip().lower() for skill in known_skills if skill})
        self.vectorizer = TfidfVectorizer(stop_words="english")

    @staticmethod
    def extract_resume_text(path: str | Path) -> str:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return text.strip()
        if path.suffix.lower() == ".txt":
            return path.read_text(encoding="utf-8").strip()
        raise ValueError("Unsupported format. Use PDF or TXT resume files.")

    @staticmethod
    def preprocess(text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"[a-zA-Z+#\.]+", text)
        return [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]

    def extract_skills(self, text: str) -> list[str]:
        tokens = set(self.preprocess(text))
        extracted = [s for s in self.known_skills if s in tokens or s.replace(" ", "") in "".join(tokens)]
        return sorted(set(extracted))

    def match_score(self, resume_text: str, job_skills: list[str]) -> float:
        if not resume_text.strip():
            return 0.0
        documents = [resume_text, " ".join(job_skills)]
        tfidf = self.vectorizer.fit_transform(documents)
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(np.round(score * 100, 2))

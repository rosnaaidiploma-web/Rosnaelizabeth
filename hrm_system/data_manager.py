from __future__ import annotations

from pathlib import Path
import pandas as pd


class DataManager:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.paths = {
            "jobs": self.data_dir / "jobs.csv",
            "candidates": self.data_dir / "candidates.csv",
            "employees": self.data_dir / "employees.csv",
            "training": self.data_dir / "training_courses.csv",
            "faq": self.data_dir / "chatbot_faq.csv",
        }

    def load(self, key: str) -> pd.DataFrame:
        return pd.read_csv(self.paths[key])

    def save(self, key: str, df: pd.DataFrame) -> None:
        df.to_csv(self.paths[key], index=False)

    def append_candidate(self, candidate: dict) -> None:
        candidates = self.load("candidates")
        updated = pd.concat([candidates, pd.DataFrame([candidate])], ignore_index=True)
        self.save("candidates", updated)

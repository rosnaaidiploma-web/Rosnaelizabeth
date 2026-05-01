from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_manager import DataManager
from .nlp_engine import ResumeAnalyzer


@dataclass
class MatchResult:
    score: float
    extracted_skills: list[str]
    missing_skills: list[str]


class HRMSystem:
    def __init__(self, data_dir: str | Path = "data", reports_dir: str | Path = "reports") -> None:
        self.dm = DataManager(data_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        jobs = self.dm.load("jobs")
        skills = set()
        for skill_list in jobs["required_skills"].fillna(""):
            skills.update(s.strip().lower() for s in str(skill_list).split(";") if s.strip())
        self.resume_analyzer = ResumeAnalyzer(skills)

    def list_jobs(self) -> pd.DataFrame:
        return self.dm.load("jobs")

    def apply_for_job(self, name: str, job_id: int, resume_path: str | Path) -> MatchResult:
        jobs = self.dm.load("jobs")
        job_row = jobs.loc[jobs["job_id"] == job_id].iloc[0]
        required_skills = [s.strip().lower() for s in job_row["required_skills"].split(";") if s.strip()]

        resume_text = self.resume_analyzer.extract_resume_text(resume_path)
        extracted = self.resume_analyzer.extract_skills(resume_text)
        missing = sorted(set(required_skills) - set(extracted))
        score = self.resume_analyzer.match_score(resume_text, required_skills)

        candidates = self.dm.load("candidates")
        candidate_id = (int(candidates["candidate_id"].max()) + 1) if not candidates.empty else 1
        self.dm.append_candidate(
            {
                "candidate_id": candidate_id,
                "name": name,
                "job_id": job_id,
                "skills": ";".join(extracted),
                "resume_text": resume_text,
                "match_score": score,
            }
        )
        return MatchResult(score=score, extracted_skills=extracted, missing_skills=missing)

    def skill_gap_report(self) -> pd.DataFrame:
        candidates = self.dm.load("candidates")
        jobs = self.dm.load("jobs")
        report_rows: list[dict[str, Any]] = []
        for _, row in candidates.iterrows():
            job = jobs.loc[jobs["job_id"] == row["job_id"]].iloc[0]
            candidate_skills = {s.strip().lower() for s in str(row["skills"]).split(";") if s.strip()}
            required_skills = {s.strip().lower() for s in str(job["required_skills"]).split(";") if s.strip()}
            missing = sorted(required_skills - candidate_skills)
            report_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "candidate_name": row["name"],
                    "job_title": job["job_title"],
                    "missing_skills": ";".join(missing),
                    "skill_gap_count": len(missing),
                    "match_score": row.get("match_score", 0),
                }
            )
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(self.reports_dir / "skill_gap_report.csv", index=False)
        return report_df

    def recommend_training(self, missing_skills: list[str]) -> pd.DataFrame:
        training = self.dm.load("training")
        rec = training[training["skill"].str.lower().isin([s.lower() for s in missing_skills])].copy()
        return rec.sort_values(["skill", "duration"])

    def promotion_analysis(self, min_exp: int = 5, min_skill_count: int = 4) -> pd.DataFrame:
        employees = self.dm.load("employees")
        employees["skill_count"] = employees["skills"].fillna("").apply(lambda x: len([s for s in str(x).split(";") if s.strip()]))
        employees["promotion_ready"] = (
            (employees["experience"] >= min_exp) & (employees["skill_count"] >= min_skill_count)
        )
        return employees.sort_values(["promotion_ready", "experience", "skill_count"], ascending=False)

    def analytics_dashboard(self) -> dict[str, str]:
        report_df = self.skill_gap_report()
        promo_df = self.promotion_analysis()

        # Common missing skills
        missing_skills = report_df["missing_skills"].str.split(";").explode()
        missing_skills = missing_skills[missing_skills.notna() & (missing_skills != "")]
        plt.figure(figsize=(10, 5))
        sns.countplot(y=missing_skills, order=missing_skills.value_counts().index)
        plt.title("Common Missing Skills")
        plt.tight_layout()
        missing_plot = self.reports_dir / "common_missing_skills.png"
        plt.savefig(missing_plot)
        plt.close()

        # Promotion status
        plt.figure(figsize=(6, 4))
        sns.countplot(x=promo_df["promotion_ready"].map({True: "Ready", False: "Not Ready"}))
        plt.title("Promotion Readiness")
        plt.tight_layout()
        promo_plot = self.reports_dir / "promotion_readiness.png"
        plt.savefig(promo_plot)
        plt.close()

        return {"missing_skills_plot": str(missing_plot), "promotion_plot": str(promo_plot)}

    def chatbot_response(self, query: str) -> str:
        faq = self.dm.load("faq")
        questions = faq["question"].tolist()
        docs = [query] + questions
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(docs)
        sims = cosine_similarity(mat[0:1], mat[1:])[0]
        best_idx = int(sims.argmax())
        if sims[best_idx] < 0.2:
            return "I couldn't find an exact answer. Please contact HR support for detailed help."
        return str(faq.iloc[best_idx]["answer"])

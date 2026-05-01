from __future__ import annotations

from pathlib import Path
from pprint import pprint

from hrm_system.hr_modules import HRMSystem


def demo() -> None:
    system = HRMSystem(data_dir="data", reports_dir="reports")

    print("\n=== JOB PORTAL ===")
    print(system.list_jobs()[["job_id", "job_title", "required_skills"]])

    sample_resume = Path("data/sample_resume.txt")
    result = system.apply_for_job(name="Alice Smith", job_id=1, resume_path=sample_resume)
    print("\n=== RESUME ANALYSIS ===")
    print(f"Match Score: {result.score}%")
    print(f"Extracted Skills: {result.extracted_skills}")
    print(f"Missing Skills: {result.missing_skills}")

    print("\n=== SKILL GAP REPORT ===")
    gap_df = system.skill_gap_report()
    print(gap_df)

    if result.missing_skills:
        print("\n=== TRAINING RECOMMENDATIONS ===")
        rec = system.recommend_training(result.missing_skills)
        print(rec[["skill", "course_name", "duration"]])

    print("\n=== PROMOTION ANALYSIS ===")
    promo = system.promotion_analysis()
    print(promo[["emp_id", "name", "experience", "skill_count", "promotion_ready"]])

    print("\n=== CHATBOT ===")
    query = "How can I get promoted?"
    print("Q:", query)
    print("A:", system.chatbot_response(query))

    print("\n=== ANALYTICS REPORTS ===")
    report_paths = system.analytics_dashboard()
    pprint(report_paths)


if __name__ == "__main__":
    demo()

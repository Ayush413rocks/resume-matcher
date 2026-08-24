"""
run_eval.py
Validates the embedding-based match score against human-judged relevance
scores on a small labeled eval set. This is the piece that turns the
project from "a demo" into "a demo with evidence it works" -- the kind
of thing worth a resume line.

Usage:
    python -m eval.run_eval
"""

import csv
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.embedder import similarity_score
from sample_data.resumes import RESUMES
from sample_data.jobs import JOBS


def load_eval_set(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run():
    eval_path = Path(__file__).parent / "eval_set.csv"
    rows = load_eval_set(eval_path)

    human_scores = []
    model_scores = []

    print(f"{'resume':<20} {'job':<24} {'human':>7} {'model':>7}")
    print("-" * 62)

    for row in rows:
        resume_text = RESUMES[row["resume_id"]]
        jd_text = JOBS[row["job_id"]]
        human = float(row["human_score"])
        model = similarity_score(resume_text, jd_text)

        human_scores.append(human)
        model_scores.append(model)

        print(f"{row['resume_id']:<20} {row['job_id']:<24} {human:>7.1f} {model:>7.1f}")

    pearson_r, _ = pearsonr(human_scores, model_scores)
    spearman_r, _ = spearmanr(human_scores, model_scores)

    print("-" * 62)
    print(f"Pearson correlation:  {pearson_r:.3f}")
    print(f"Spearman correlation: {spearman_r:.3f}")
    print(
        "\nInterpretation: correlation above ~0.7 suggests the embedding "
        "score is a reasonable proxy for human relevance judgment on this "
        "eval set. Expand eval_set.csv with more diverse, real pairs "
        "before citing this number anywhere serious."
    )


if __name__ == "__main__":
    run()

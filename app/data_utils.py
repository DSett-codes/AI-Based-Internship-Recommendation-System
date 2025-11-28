"""Helpers for loading and cleaning the internship dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "AI-based Career Recommendation System.csv"


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the career recommendation dataset and normalize column names."""
    data_path = Path(path) if path else DATASET_PATH
    df = pd.read_csv(data_path)
    df = df.rename(
        columns={
            "CandidateID": "candidate_id",
            "Name": "name",
            "Age": "age",
            "Education": "education",
            "Skills": "skills",
            "Interests": "interests",
            "Recommended_Career": "recommended_career",
            "Recommendation_Score": "recommendation_score",
        }
    )
    # Normalize text columns for easier matching
    for col in ["skills", "interests", "education"]:
        if col in df:
            df[col] = df[col].astype(str).str.strip()
    return df

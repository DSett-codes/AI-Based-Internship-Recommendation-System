"""Tiny ML helper used by the Flask app.

This file intentionally keeps the pipeline simple so new ML learners
can follow along. The model trains on the CSV each time the server
starts and stays in memory while Flask is running.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .data_utils import load_dataset

DATASET_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "AI-based Career Recommendation System.csv"


def combine_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Create a single text column from skills, interests, and education."""
    df = df.copy()
    df["text"] = (
        df["education"].fillna("")
        + " "
        + df["skills"].fillna("")
        + " "
        + df["interests"].fillna("")
    )
    return df


def build_pipeline() -> Pipeline:
    """Simple text model using TF-IDF + Logistic Regression."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
            ("clf", HistGradientBoostingClassifier(max_iter=300)),
        ]
    )


def train_model(dataset_path: Path | None = None) -> Tuple[Pipeline, List[str]]:
    df = load_dataset(dataset_path or DATASET_PATH)
    df = combine_text_fields(df)

    model = build_pipeline()
    model.fit(df["text"], df["recommended_career"])
    return model, sorted(df["recommended_career"].unique())


def predict_top_careers(model: Pipeline, careers: List[str], profile: dict, top_n: int = 3):
    """Return top careers with probabilities for the given profile."""
    profile_text = f"{profile.get('education', '')} {profile.get('skills', '')} {profile.get('interests', '')}"
    probs = model.predict_proba([profile_text])[0]

    ranked = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {"career": career, "probability": round(prob, 3)}
        for career, prob in ranked
    ]

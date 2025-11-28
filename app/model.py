"""Model utilities for training a lightweight transformer-style pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .data_utils import load_dataset


MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "pilot_transformer.joblib"


def build_feature_pipeline() -> ColumnTransformer:
    """Create the feature transformer for tabular + text inputs."""
    text_params = dict(ngram_range=(1, 2), min_df=2, max_features=400)

    skills_vectorizer = TfidfVectorizer(**text_params)
    interests_vectorizer = TfidfVectorizer(**text_params)

    transformer = ColumnTransformer(
        transformers=[
            (
                "skills",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("tfidf", skills_vectorizer),
                    ]
                ),
                "skills",
            ),
            (
                "interests",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("tfidf", interests_vectorizer),
                    ]
                ),
                "interests",
            ),
            (
                "education",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["education"],
            ),
            (
                "age",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["age"],
            ),
        ],
        remainder="drop",
    )
    return transformer


def build_model() -> Pipeline:
    """Assemble the full pipeline with a multinomial logistic regression."""
    features = build_feature_pipeline()
    classifier = LogisticRegression(max_iter=500, multi_class="auto")
    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


def train_model(df: pd.DataFrame) -> Pipeline:
    model = build_model()
    X = df[["skills", "interests", "education", "age"]]
    y = df["recommended_career"]
    model.fit(X, y)
    return model


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_or_train_model(dataset_path: Path = None, model_path: Path = MODEL_PATH) -> Pipeline:
    model_file = Path(model_path)
    if model_file.exists():
        return joblib.load(model_file)

    df = load_dataset(dataset_path)
    model = train_model(df)
    save_model(model, model_file)
    return model


@dataclass
class RecommendationResult:
    career: str
    probability: float
    rationale: str


class RecommendationEngine:
    """Hybrid rule-based + ML-light scoring for internship recommendations."""

    def __init__(self, dataset_path: Path = None, model_path: Path = MODEL_PATH):
        self.dataset = load_dataset(dataset_path)
        self.model = load_or_train_model(dataset_path, model_path)
        self.unique_careers: List[str] = list(self.dataset["recommended_career"].unique())

    @staticmethod
    def _clean_list(text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return [token.strip().lower() for token in text.replace(",", ";").split(";") if token.strip()]

    def _rule_based_boost(self, profile: dict, career: str) -> float:
        record = self.dataset[self.dataset["recommended_career"] == career].iloc[0]
        profile_skills = set(self._clean_list(profile.get("skills", "")))
        profile_interests = set(self._clean_list(profile.get("interests", "")))
        career_skills = set(self._clean_list(record.get("skills", "")))
        career_interests = set(self._clean_list(record.get("interests", "")))

        skill_overlap = len(profile_skills & career_skills) / max(len(career_skills), 1)
        interest_overlap = len(profile_interests & career_interests) / max(len(career_interests), 1)

        education_match = 1.0 if str(profile.get("education", "")).lower() == str(record.get("education", "")).lower() else 0.0
        return 0.1 * skill_overlap + 0.05 * interest_overlap + 0.05 * education_match

    def recommend(self, profile: dict, top_n: int = 3) -> List[RecommendationResult]:
        input_df = pd.DataFrame([
            {
                "skills": profile.get("skills", ""),
                "interests": profile.get("interests", ""),
                "education": profile.get("education", ""),
                "age": profile.get("age", None),
            }
        ])

        proba = self.model.predict_proba(input_df)[0]
        ranked = sorted(zip(self.model.classes_, proba), key=lambda x: x[1], reverse=True)

        results: List[RecommendationResult] = []
        for career, base_prob in ranked[:top_n]:
            boost = self._rule_based_boost(profile, career)
            final_prob = min(base_prob + boost, 1.0)
            rationale = self._build_rationale(profile, career, base_prob, boost)
            results.append(RecommendationResult(career=career, probability=final_prob, rationale=rationale))
        return results

    def _build_rationale(self, profile: dict, career: str, base_prob: float, boost: float) -> str:
        parts: List[str] = [f"Model score: {base_prob:.2f}"]
        if boost > 0:
            parts.append(f"Rule-based alignment boost: +{boost:.2f}")
        matched_skills = self._describe_overlap(profile, career)
        if matched_skills:
            parts.append(f"Overlap on skills/interests: {matched_skills}")
        return "; ".join(parts)

    def _describe_overlap(self, profile: dict, career: str) -> str:
        record = self.dataset[self.dataset["recommended_career"] == career].iloc[0]
        profile_skills = set(self._clean_list(profile.get("skills", "")))
        profile_interests = set(self._clean_list(profile.get("interests", "")))
        career_skills = set(self._clean_list(record.get("skills", "")))
        career_interests = set(self._clean_list(record.get("interests", "")))

        matched_skills = profile_skills & career_skills
        matched_interests = profile_interests & career_interests

        details = []
        if matched_skills:
            details.append(f"skills: {', '.join(sorted(matched_skills))}")
        if matched_interests:
            details.append(f"interests: {', '.join(sorted(matched_interests))}")
        return "; ".join(details)

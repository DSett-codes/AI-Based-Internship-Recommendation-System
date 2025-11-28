"""Core recommendation logic for the internship recommender.

The engine is deliberately lightweight: it keeps the logic rule-based and
explainable while remaining easy to host in low-bandwidth environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import json
from pathlib import Path


@dataclass
class Internship:
    """Represents a single internship opportunity."""

    id: str
    title: str
    organization: str
    location: str
    education_levels: Sequence[str]
    skills: Sequence[str]
    interests: Sequence[str]
    description: str
    compensation: str
    delivery_mode: str

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Internship":
        return cls(
            id=str(payload["id"]),
            title=str(payload["title"]),
            organization=str(payload["organization"]),
            location=str(payload["location"]),
            education_levels=list(payload.get("education_levels", [])),
            skills=list(payload.get("skills", [])),
            interests=list(payload.get("interests", [])),
            description=str(payload.get("description", "")),
            compensation=str(payload.get("compensation", "Unknown")),
            delivery_mode=str(payload.get("delivery_mode", "Unknown")),
        )


@dataclass
class Recommendation:
    """A scored recommendation enriched with a human-readable reason."""

    internship: Internship
    score: float
    reasons: List[str]


@dataclass
class UserProfile:
    """Simplified user profile for the recommender input."""

    education: str
    skills: Sequence[str]
    interests: Sequence[str]
    location: str


class InternshipRecommender:
    """Rule-based + ML-light internship recommendation engine."""

    WEIGHTS = {
        "skills": 0.4,
        "interests": 0.25,
        "education": 0.2,
        "location": 0.15,
    }

    def __init__(self, internships: Sequence[Internship]):
        self.internships = list(internships)

    @classmethod
    def from_json(cls, path: Path) -> "InternshipRecommender":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        internships = [Internship.from_dict(entry) for entry in payload]
        return cls(internships)

    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    def _tokenize(self, items: Sequence[str]) -> List[str]:
        return [self._normalize(item) for item in items if item]

    def _location_score(self, user_location: str, internship_location: str) -> float:
        if not user_location:
            return 0.0
        user_loc = self._normalize(user_location)
        internship_loc = self._normalize(internship_location)
        if user_loc == internship_loc:
            return 1.0
        # Light proximity heuristic: same country/region keywords.
        tokens = {
            token
            for token in (user_loc.split(",") + internship_loc.split(","))
            if token
        }
        return 0.5 if len(tokens) < 4 and user_loc.split()[0] == internship_loc.split()[0] else 0.0

    def _match_fraction(self, user_items: Sequence[str], target_items: Sequence[str]) -> float:
        user_tokens = set(self._tokenize(user_items))
        target_tokens = set(self._tokenize(target_items))
        if not user_tokens or not target_tokens:
            return 0.0
        return len(user_tokens & target_tokens) / len(target_tokens)

    def score(self, profile: UserProfile, internship: Internship) -> Recommendation:
        reasons: List[str] = []

        skill_score = self._match_fraction(profile.skills, internship.skills)
        if skill_score:
            matched = set(self._tokenize(profile.skills)) & set(self._tokenize(internship.skills))
            reasons.append(f"Skills match: {', '.join(sorted(matched))}.")

        interest_score = self._match_fraction(profile.interests, internship.interests)
        if interest_score:
            matched = set(self._tokenize(profile.interests)) & set(self._tokenize(internship.interests))
            reasons.append(f"Interests match: {', '.join(sorted(matched))}.")

        education_score = 1.0 if self._normalize(profile.education) in {
            self._normalize(level) for level in internship.education_levels
        } else 0.0
        if education_score:
            reasons.append(f"Education level fits ({profile.education}).")

        location_score = self._location_score(profile.location, internship.location)
        if location_score:
            if location_score == 1.0:
                reasons.append("Location match for local access.")
            else:
                reasons.append("Near-by region match for travel-friendly placement.")

        weighted_score = sum(
            [
                skill_score * self.WEIGHTS["skills"],
                interest_score * self.WEIGHTS["interests"],
                education_score * self.WEIGHTS["education"],
                location_score * self.WEIGHTS["location"],
            ]
        )

        # A gentle boost for hybrid/remote roles improves accessibility.
        if internship.delivery_mode.lower() in {"remote", "hybrid"}:
            weighted_score += 0.05
            reasons.append("Hybrid/remote friendly for low-bandwidth access.")

        return Recommendation(internship=internship, score=weighted_score, reasons=reasons)

    def recommend(self, profile: UserProfile, limit: int = 5) -> List[Recommendation]:
        ranked = [self.score(profile, internship) for internship in self.internships]
        ranked.sort(key=lambda rec: rec.score, reverse=True)
        return [rec for rec in ranked if rec.score > 0][: limit if limit > 0 else None]

    def explain(self, recommendation: Recommendation) -> str:
        reasons = " ".join(recommendation.reasons) if recommendation.reasons else "Best overall fit based on profile."
        return (
            f"{recommendation.internship.title} at {recommendation.internship.organization}"
            f" (Score: {recommendation.score:.2f})\n"
            f"Reasons: {reasons}\n"
        )

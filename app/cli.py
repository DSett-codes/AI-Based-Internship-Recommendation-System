"""Command-line entry point for the internship recommender."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.recommendation_engine import InternshipRecommender, UserProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-based internship recommendation prototype")
    parser.add_argument("--education", required=True, help="Highest completed education level (e.g., Diploma, Bachelor)")
    parser.add_argument("--skills", required=True, help="Comma-separated skills")
    parser.add_argument("--interests", required=True, help="Comma-separated interests")
    parser.add_argument("--location", required=True, help="City or region")
    parser.add_argument("--data", default=Path(__file__).with_name("internships_data.json"), type=Path, help="Path to internship dataset JSON")
    parser.add_argument("--limit", type=int, default=5, help="Number of recommendations to show")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommender = InternshipRecommender.from_json(args.data)
    profile = UserProfile(
        education=args.education,
        skills=[s.strip() for s in args.skills.split(",")],
        interests=[i.strip() for i in args.interests.split(",")],
        location=args.location,
    )

    recommendations = recommender.recommend(profile, limit=args.limit)
    if not recommendations:
        print("No matches found. Try broadening your skills or location keywords.")
        return

    for rec in recommendations:
        print(recommender.explain(rec))


if __name__ == "__main__":
    main()

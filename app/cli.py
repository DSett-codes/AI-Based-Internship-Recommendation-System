"""Command-line interface for requesting internship recommendations."""
from __future__ import annotations

import argparse
from .model import RecommendationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get top internship recommendations.")
    parser.add_argument("--education", required=True, help="Education level, e.g., Bachelor's")
    parser.add_argument("--skills", required=True, help="Comma- or semicolon-separated skills")
    parser.add_argument("--interests", required=True, help="Comma- or semicolon-separated interests")
    parser.add_argument("--age", type=int, default=None, help="Age (optional)")
    parser.add_argument("--limit", type=int, default=3, help="Number of recommendations to return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = RecommendationEngine()
    profile = {
        "education": args.education,
        "skills": args.skills,
        "interests": args.interests,
        "age": args.age,
    }
    results = engine.recommend(profile, top_n=args.limit)
    print("Top internship-style recommendations:\n")
    for idx, rec in enumerate(results, start=1):
        print(f"{idx}. {rec.career} — {rec.probability:.2f}")
        print(f"   Why: {rec.rationale}\n")


if __name__ == "__main__":
    main()

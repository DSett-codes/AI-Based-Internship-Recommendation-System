# AI-Based Internship Recommendation System (Hackathon Prototype)

Lightweight, explainable recommendation engine tailored for first-generation and under-served learners. The prototype mixes transparent rule-based logic with a tiny ML-style scoring heuristic so it can run on low-bandwidth, mobile-first government portals.

## Features
- **Inputs:** education level, skills, interests, and location.
- **Outputs:** Top 3–5 personalized internships with plain-language reasons.
- **Architecture:** Rule-based scoring with a small accessibility boost for remote/hybrid roles—easy to audit and integrate.
- **Deployment:** Pure Python, no external services; the JSON dataset can be swapped for live feeds later.

## Quickstart
1. Ensure Python 3.9+ is available.
2. Run the CLI with your profile details:
   ```bash
   python -m app.cli \
     --education "Bachelor" \
     --skills "python, data analysis" \
     --interests "analytics, ai" \
     --location "Nairobi" \
     --limit 4
   ```
3. The tool returns ranked internships with short explanations that can be shown on mobile or SMS-friendly UI layers.

## How the scoring works
The engine favors transparency and explainability:
- **Skills (40%)** – overlap between the user’s skills and the internship needs.
- **Interests (25%)** – alignment with the internship focus areas.
- **Education (20%)** – binary match on allowed education levels.
- **Location (15%)** – exact or nearby-region match to reduce travel costs.
- **Accessibility boost (+0.05)** – for hybrid/remote roles suitable for low-bandwidth access.

Each recommendation includes the matched factors so program officers can spot-fit issues quickly.

## Extending the prototype
- Replace `app/internships_data.json` with live data from government or NGO feeds.
- Adjust weights in `InternshipRecommender.WEIGHTS` to tune priorities per region.
- Wrap `app/recommendation_engine.py` with a FastAPI or Flask endpoint for portal integration.

## Repository layout
- `app/recommendation_engine.py` – core logic and explainability helpers.
- `app/cli.py` – minimal command-line interface.
- `app/internships_data.json` – sample dataset for testing.

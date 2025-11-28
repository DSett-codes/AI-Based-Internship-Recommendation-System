# AI-Based Internship Recommendation System (Hackathon Prototype)

A lightweight, explainable recommendation engine designed for first-generation and under-served learners. It blends transparent rule-based logic with a compact ML pipeline so it can run on low-bandwidth, mobile-first government portals.

## Features
- **Inputs:** education level, skills, interests, and optional age.
- **Outputs:** Top 3–5 internship-style career recommendations with clear rationales.
- **Architecture:** Rule-based scoring layered on top of a multinomial logistic regression trained on the provided CSV dataset (via a `ColumnTransformer` feature builder).
- **Deployment:** Pure Python + Flask; no external services. Designed for easy integration with portals or SMS layers.

## Dataset
- Stored at `artifacts/AI-based Career Recommendation System.csv`.
- Columns include `Education`, `Skills`, `Interests`, and `Recommended_Career` with historical recommendation scores.

## Project layout
- `app/data_utils.py` – dataset loader/normalizer.
- `app/model.py` – hybrid rule-based + ML pipeline and recommenders.
- `app/training.py` – CLI to train and persist `artifacts/pilot_transformer.joblib`.
- `app/cli.py` – minimal CLI for top-N recommendations.
- `app/server.py` – Flask app exposing JSON (`/api/recommend`) and HTML form (`/`).
- `artifacts/` – dataset and model artifacts.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the pilot model
```bash
python -m app.training --model-path artifacts/pilot_transformer.joblib
```
The CLI will train from the CSV in `artifacts/` (or `--dataset` to override) and persist a ready-to-serve model.

## Use the CLI
```bash
python -m app.cli \
  --education "Bachelor's" \
  --skills "python; data analysis; machine learning" \
  --interests "ai; analytics" \
  --age 25 \
  --limit 3
```
The CLI prints the top-ranked careers with probabilities and rule-based rationales.

## Run the Flask app
```bash
export FLASK_APP=app.server:app
flask run --host 0.0.0.0 --port 8000
```
- Visit `http://localhost:8000` for the mobile-friendly form.
- POST JSON to `/api/recommend` with keys `education`, `skills`, `interests`, and optional `age` (query param `limit` to control result count).

## How the scoring works
- **Skills & interests (TF–IDF):** text features vectorized via `ColumnTransformer` and logistic regression to estimate base probabilities.
- **Education & age:** categorical + numeric features inform the model alongside text.
- **Rule-based boost:** small additive bonus for overlapping skills/interests and exact education match to keep results explainable.

This hybrid keeps the model auditable while still adapting to the dataset patterns for fair, transparent internship recommendations.

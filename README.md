# AI-Based Internship Recommendation System (Beginner Flask Demo)

This is a very small machine learning web app built with Flask. When the server
starts, it trains a TF-IDF + Logistic Regression model on the provided dataset
and keeps everything in memory. Enter your education, skills, and interests in
the form to see the top matching internship-style careers.

## How it works
- Dataset lives in `artifacts/AI-based Career Recommendation System.csv`.
- On startup the app combines the education, skills, and interests columns into
  one text field.
- A TF-IDF vectorizer feeds a logistic regression classifier to predict the
  recommended career.
- The Flask routes render an HTML form at `/` and also expose a JSON endpoint at
  `/api/recommend`.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export FLASK_APP=app.server:app
flask run --host 0.0.0.0 --port 8000
```
Then open [http://localhost:8000](http://localhost:8000) and submit the form.

## Request JSON directly
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "education": "Bachelor's", 
    "skills": "python, data analysis", 
    "interests": "ai, analytics"
  }'
```

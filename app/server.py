"""Minimal Flask app exposing the recommendation engine."""
from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from .model import RecommendationEngine

app = Flask(__name__)
engine = RecommendationEngine()

FORM_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Internship Recommender</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 780px; margin: auto; padding: 1rem; }
    label { display: block; margin-top: 0.5rem; font-weight: bold; }
    input, textarea { width: 100%; padding: 0.4rem; margin-top: 0.25rem; }
    button { margin-top: 0.75rem; padding: 0.5rem 1rem; }
    .result { margin-top: 1rem; padding: 0.75rem; border: 1px solid #ddd; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>AI-Based Internship Recommendation</h1>
  <form method="post">
    <label>Education</label>
    <input type="text" name="education" required placeholder="Bachelor's" />

    <label>Skills</label>
    <textarea name="skills" required placeholder="python; data analysis; machine learning"></textarea>

    <label>Interests</label>
    <textarea name="interests" required placeholder="ai; analytics"></textarea>

    <label>Age (optional)</label>
    <input type="number" name="age" />

    <label>How many results?</label>
    <input type="number" name="limit" value="3" min="1" max="5" />

    <button type="submit">Recommend internships</button>
  </form>

  {% if recommendations %}
    <h2>Top recommendations</h2>
    {% for rec in recommendations %}
      <div class="result">
        <strong>{{ loop.index }}. {{ rec.career }}</strong> — score {{ '%.2f' % rec.probability }}<br/>
        <small>{{ rec.rationale }}</small>
      </div>
    {% endfor %}
  {% endif %}
</body>
</html>
"""


def parse_request_data(req) -> dict:
    data = req.get_json(silent=True) or {}
    if not data:
        data = req.form.to_dict()
    return {
        "education": data.get("education", ""),
        "skills": data.get("skills", ""),
        "interests": data.get("interests", ""),
        "age": int(data["age"]) if data.get("age") else None,
    }


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    payload = parse_request_data(request)
    limit = int(request.args.get("limit", request.form.get("limit", 3)))
    results = engine.recommend(payload, top_n=limit)
    return jsonify(
        [
            {"career": r.career, "probability": r.probability, "rationale": r.rationale}
            for r in results
        ]
    )


@app.route("/", methods=["GET", "POST"])
def form():
    recommendations = None
    if request.method == "POST":
        payload = parse_request_data(request)
        limit = int(request.form.get("limit", 3))
        recommendations = engine.recommend(payload, top_n=limit)
    return render_template_string(FORM_TEMPLATE, recommendations=recommendations)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

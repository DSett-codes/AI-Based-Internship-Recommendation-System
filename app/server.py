"""Simple Flask app that trains a tiny model and serves predictions."""
from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from .model import predict_top_careers, train_model

app = Flask(__name__)

# Train once when the server starts so requests are fast.
MODEL, CAREERS = train_model()


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Internship Recommender</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 720px; margin: auto; padding: 1rem; }
    form { display: grid; gap: 0.5rem; }
    label { font-weight: bold; }
    textarea, input { width: 100%; padding: 0.45rem; }
    button { padding: 0.6rem 0.9rem; font-size: 1rem; }
    .card { margin-top: 1rem; padding: 0.75rem; border: 1px solid #ddd; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>AI Internship Recommendation (Beginner Version)</h1>
  <p>This demo trains a small TF-IDF + Logistic Regression model on the dataset when the app starts.</p>
  <form method="post">
    <label>Education level</label>
    <input type="text" name="education" required placeholder="Bachelor's in Computer Science" />

    <label>Skills (separate with commas)</label>
    <textarea name="skills" required placeholder="python, data analysis, machine learning"></textarea>

    <label>Interests</label>
    <textarea name="interests" required placeholder="ai, analytics"></textarea>

    <label>How many results?</label>
    <input type="number" name="limit" value="3" min="1" max="5" />

    <button type="submit">Get recommendations</button>
  </form>

  {% if recommendations %}
    <h2>Top matches</h2>
    {% for rec in recommendations %}
      <div class="card">
        <strong>{{ loop.index }}. {{ rec.career }}</strong><br/>
        Probability: {{ '%.2f' % rec.probability }}
      </div>
    {% endfor %}
  {% endif %}
</body>
</html>
"""


def parse_payload(req) -> dict:
    data = req.get_json(silent=True) or req.form.to_dict()
    return {
        "education": data.get("education", ""),
        "skills": data.get("skills", ""),
        "interests": data.get("interests", ""),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    if request.method == "POST":
        payload = parse_payload(request)
        limit = int(request.form.get("limit", 3))
        recommendations = predict_top_careers(MODEL, CAREERS, payload, top_n=limit)
    return render_template_string(PAGE, recommendations=recommendations)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    payload = parse_payload(request)
    limit = int(request.args.get("limit", request.form.get("limit", 3)))
    results = predict_top_careers(MODEL, CAREERS, payload, top_n=limit)
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

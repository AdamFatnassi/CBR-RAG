"""
rag_server.py : Tiny local server bridging the static web demo to the Groq LLM API.

The web demo (web/index.html) is a fully offline, static page – it cannot call Groq
directly because the API key must stay secret.  This server solves that by running
locally alongside the browser tab.

Usage (from the project root):
    .venv\\Scripts\\python web\\rag_server.py

Then open web/index.html, fill in a client, and click "Generate AI explanation".
The button will POST the current client + its retrieved neighbours to this server,
which builds the prompt (augmentation.py) and calls Groq (rag_generator.py).

Endpoint
--------
POST  http://localhost:5050/explain
Body  (JSON):
  {
    "query_data":       { "age": 38, "job": "admin.", ... },   // raw client values
    "prediction":       0,                                      // 0 or 1
    "score":            0.033,                                  // weighted share of yes
    "retrieved_cases":  [                                       // top-M cases
      { "rank": 1, "similarity": 0.931, "outcome": "no",
        "age": 38, "job": "admin.", ... },
      ...
    ]
  }
Response (JSON):
  { "explanation": "<LLM text>", "decision": "no", "model": "<model name>" }
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify          # noqa: E402
from src.rag_generator import CBR_RAG_Generator    # noqa: E402

app = Flask(__name__)
PORT = 5050

# ── CORS: allow the file:// origin used when opening index.html directly ──────
@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


# ── Lazy-load the generator once ──────────────────────────────────────────────
_generator: CBR_RAG_Generator | None = None

def get_generator() -> CBR_RAG_Generator:
    global _generator
    if _generator is None:
        _generator = CBR_RAG_Generator(use_api=True)
    return _generator


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    gen = get_generator()
    return jsonify({
        "status": "ok",
        "api": gen.use_api,
        "model": gen.model_name,
        "message": "RAG server is running. POST /explain to generate an explanation."
    })


@app.route("/explain", methods=["POST", "OPTIONS"])
def explain():
    if request.method == "OPTIONS":      # pre-flight
        return "", 204

    context = request.get_json(force=True, silent=True)
    if not context:
        return jsonify({"error": "Empty or invalid JSON body"}), 400

    try:
        gen = get_generator()
        explanation = gen.generate_explanation(context)
        return jsonify({
            "explanation": explanation,
            "decision": "yes" if context.get("prediction") == 1 else "no",
            "model": gen.model_name,
            "api": gen.use_api,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  RAG server  →  http://localhost:{PORT}")
    print(f"  Health     →  http://localhost:{PORT}/")
    print(f"  Endpoint   →  POST http://localhost:{PORT}/explain")
    print()
    print("  Open web/index.html in your browser,")
    print("  then click  ✦ Generate AI explanation")
    print("=" * 60)
    app.run(host="127.0.0.1", port=PORT, debug=False)

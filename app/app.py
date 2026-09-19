# app.py
import base64
import os
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify, session

from config import (
    OLLAMA_MODEL, TEMPERATURE, TOP_P, NUM_CTX,
    CAAP_ENABLED, ISTM_ENABLED,
    CAAP_ANCHOR_PROMPT,
    ISTM_PRESSURE_MARKERS,
    SYSTEM_PROMPT,
)

import ollama

app = Flask(__name__)
app.secret_key = "vlm-no-sycophancy-secret-key-change-me"

# ---------------------------------------------------------------------------
# Server-side store — keyed by a small session id.
# This avoids putting the (huge) base64 image inside the Flask cookie.
# ---------------------------------------------------------------------------
STORE = {}   # { sid: {"image_b64": str, "history": list, "anchor": str|None} }
TEMP_DIR = tempfile.mkdtemp(prefix="vlm_uploads_")


def get_session_id() -> str:
    """Return (and create if missing) a lightweight session id."""
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def get_store() -> dict:
    sid = get_session_id()
    if sid not in STORE:
        STORE[sid] = {"image_b64": None, "history": [], "anchor": None}
    return STORE[sid]


# ---------------------------------------------------------------------------
# ISTM helpers
# ---------------------------------------------------------------------------
def detect_pressure(question: str) -> bool:
    q = question.lower().strip()
    return any(marker in q for marker in ISTM_PRESSURE_MARKERS)


# ---------------------------------------------------------------------------
# CAAP
# ---------------------------------------------------------------------------
def build_visual_anchor(image_b64: str) -> str:
    if not CAAP_ENABLED:
        return ""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": CAAP_ANCHOR_PROMPT,
                "images": [image_b64],
            }],
            options={"temperature": 0.0, "top_p": 1.0, "num_ctx": NUM_CTX},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[CAAP] Anchor extraction failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# ISTM
# ---------------------------------------------------------------------------
def build_istm_guard(question: str, anchor: str) -> str:
    if not ISTM_ENABLED:
        return ""

    if detect_pressure(question):
        guard = (
            "SYCOPHANCY ALERT: The user's message contains a leading or biased claim. "
            "You MUST NOT agree with it unless the visual evidence genuinely supports it. "
            "Compare the user's claim directly against the visual anchor description. "
            "If they conflict, state the truth clearly and correct the user.\n\n"
        )
    else:
        guard = (
            "ISTM GUARD: Answer neutrally. Do not over-align with user phrasing. "
            "Ground every claim in the visual anchor.\n\n"
        )

    if anchor:
        guard += f"VISUAL ANCHOR (objective description):\n{anchor}\n\n"

    return guard


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------
def build_messages(question: str, image_b64: str, history: list, anchor: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    guard = build_istm_guard(question, anchor)

    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": f"{guard}USER QUESTION: {question}",
        "images": [image_b64],
    })
    return messages


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Store image server-side. No analysis yet."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    raw = file.read()
    b64 = base64.b64encode(raw).decode("utf-8")

    store = get_store()
    store["image_b64"] = b64
    store["history"] = []
    store["anchor"] = None

    print(f"[UPLOAD] sid={session.get('sid')} bytes={len(raw)}")

    return jsonify({
        "status": "ok",
        "preview": f"data:image/jpeg;base64,{b64}",
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    store = get_store()
    image_b64 = store.get("image_b64")

    print(f"[ASK] sid={session.get('sid')} has_image={bool(image_b64)} q={question!r}")

    if not image_b64:
        return jsonify({"error": "No image uploaded. Please upload an image first."}), 400

    history = store.get("history", [])
    anchor = store.get("anchor")

    # CAAP: compute anchor once per image
    if anchor is None and CAAP_ENABLED:
        anchor = build_visual_anchor(image_b64)
        store["anchor"] = anchor
        print(f"[CAAP] anchor built ({len(anchor)} chars)")

    messages = build_messages(question, image_b64, history, anchor or "")

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "num_ctx": NUM_CTX,
            },
        )
        answer = response["message"]["content"].strip()
    except Exception as e:
        return jsonify({"error": f"Ollama error: {str(e)}"}), 500

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    store["history"] = history

    return jsonify({
        "answer": answer,
        "anchor": anchor,
        "pressure_detected": detect_pressure(question),
    })


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid and sid in STORE:
        del STORE[sid]
    session.pop("sid", None)
    return jsonify({"status": "reset"})


@app.route("/debug")
def debug():
    """Handy endpoint to check server state from the browser."""
    sid = session.get("sid")
    store = STORE.get(sid, {}) if sid else {}
    return jsonify({
        "sid": sid,
        "has_image": bool(store.get("image_b64")),
        "history_len": len(store.get("history", [])),
        "anchor_len": len(store.get("anchor") or ""),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
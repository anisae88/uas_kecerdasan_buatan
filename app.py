import json
import re
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, url_for

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)
app.secret_key = "praktikum-cnn-tokopedia"

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
PLOTS_DIR = Path("static") / "plots"
REPORT_PATH = DATA_DIR / "hasil.json"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
MODEL_PATH = MODEL_DIR / "cnn_tokopedia.keras"
MAX_LEN = 120


def clean_text(text):
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_report():
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "accuracy": None,
        "loss": None,
        "class_names": [],
        "classification_report_text": "Jalankan cnn.py terlebih dahulu.",
        "plots": [],
    }


def load_predictor():
    if not (MODEL_PATH.exists() and TOKENIZER_PATH.exists() and LABEL_ENCODER_PATH.exists()):
        return None, None, None
    from keras.models import load_model
    model = load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, tokenizer, label_encoder


def get_plot_items():
    report = load_report()
    plots = []
    for item in report.get("plots", []):
        file = item.get("file", "")
        plots.append({
            "title": item.get("title", "Grafik"),
            "description": item.get("description", ""),
            "path": url_for("static", filename=file.replace("plots/", "plots/")),
        })
    return plots


@app.route("/", methods=["GET"])
def index():
    report = load_report()
    predictor = load_predictor()
    available = all(predictor)
    return render_template(
        "index.html",
        report=report,
        model_ready=available,
    )


@app.route("/hasil", methods=["GET"])
def hasil():
    report = load_report()
    plots = get_plot_items()
    return render_template("hasil.html", report=report, plots=plots)


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        flash("Masukkan teks review terlebih dahulu.", "warning")
        return redirect(url_for("index"))

    model, tokenizer, label_encoder = load_predictor()
    if model is None:
        flash("Model belum tersedia. Jalankan cnn.py terlebih dahulu.", "danger")
        return redirect(url_for("index"))

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(prob))
    label = label_encoder.inverse_transform([idx])[0]
    confidence = float(prob[idx])

    report = load_report()
    plots = get_plot_items()

    return render_template(
        "hasil.html",
        report=report,
        plots=plots,
        prediction={
            "text": text,
            "cleaned": cleaned,
            "label": label,
            "confidence": confidence,
        },
    )


if __name__ == "__main__":
    app.run(debug=True)

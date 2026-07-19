import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "static" / "plots"
MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = DATA_DIR / "hasil.json"
MANIFEST_PATH = DATA_DIR / "plot_manifest.json"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
MODEL_PATH = MODEL_DIR / "cnn_tokopedia.keras"

MAX_WORDS = 12000
MAX_LEN = 120
EMBED_DIM = 128
BATCH_SIZE = 32
EPOCHS = 8
TEST_SIZE = 0.2
RANDOM_STATE = 42


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_column(columns, candidates):
    normalized = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in normalized:
            return normalized[key]
    for col in columns:
        s = str(col).lower()
        for cand in candidates:
            if cand.lower() in s:
                return col
    return None


def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_sentiment(rating):
    try:
        r = float(rating)
    except Exception:
        return np.nan
    if r <= 2:
        return "Negatif"
    elif r == 3:
        return "Netral"
    return "Positif"


def detect_dataset_file():
    candidates = [
        DATA_DIR / "tokopedia_reviews.csv",
        DATA_DIR / "tokopedia_reviews_raw.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("File data/tokopedia_reviews.csv belum ditemukan. Jalankan ambil_data.py terlebih dahulu.")


def save_plot(fig, filename):
    safe_mkdir(PLOTS_DIR)
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def add_manifest_item(manifest, filename, title, description):
    manifest.append({
        "file": filename,
        "title": title,
        "description": description
    })


def plot_histogram(series, title, xlabel, filename, bins=30):
    fig = plt.figure(figsize=(9, 5))
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frekuensi")
    return save_plot(fig, filename)


def plot_bar(series, title, xlabel, ylabel, filename):
    fig = plt.figure(figsize=(9, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return save_plot(fig, filename)


def top_words(texts, n=20):
    counter = Counter()
    for text in texts:
        counter.update([w for w in str(text).split() if len(w) > 2])
    return counter.most_common(n)


def plot_top_words(texts, title, filename, n=20):
    items = top_words(texts, n=n)
    if not items:
        items = [("kosong", 1)]
    words, counts = zip(*items)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Frekuensi")
    return save_plot(fig, filename)


def plot_wordcloud(texts, title, filename):
    text = " ".join([str(t) for t in texts if pd.notna(t)])
    if not text.strip():
        text = "kosong"
    wc = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(text)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    return save_plot(fig, filename)


def plot_confusion_matrix(cm, labels, filename):
    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    return save_plot(fig, filename)


def plot_history(history):
    fig1 = plt.figure(figsize=(9, 5))
    plt.plot(history.history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    plt.title("Grafik Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_path = save_plot(fig1, "cnn_accuracy.png")

    fig2 = plt.figure(figsize=(9, 5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Grafik Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_path = save_plot(fig2, "cnn_loss.png")
    return acc_path, loss_path


def build_model(num_classes):
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        tf.keras.layers.SpatialDropout1D(0.2),
        Conv1D(128, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def maybe_balance_minimum(df, label_col, min_count=250):
    counts = df[label_col].value_counts()
    target = min(counts.min(), min_count)
    frames = []
    for label, grp in df.groupby(label_col):
        if len(grp) > target:
            frames.append(grp.sample(target, random_state=RANDOM_STATE))
        else:
            frames.append(grp)
    return pd.concat(frames, axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main():
    safe_mkdir(DATA_DIR)
    safe_mkdir(PLOTS_DIR)
    safe_mkdir(MODEL_DIR)

    data_file = detect_dataset_file()
    df = pd.read_csv(data_file)

    rating_col = detect_column(df.columns, ["rating", "score", "stars", "bintang", "nilai"])
    text_col = detect_column(df.columns, ["review", "content", "text", "ulasan", "comment", "komentar", "message"])

    if rating_col is None or text_col is None:
        raise SystemExit(
            f"Kolom rating/teks belum terdeteksi otomatis. Kolom tersedia: {list(df.columns)}"
        )

    print("Kolom rating:", rating_col)
    print("Kolom teks:", text_col)

    df = df[[rating_col, text_col]].dropna().copy()
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col, text_col]).copy()
    df["sentimen"] = df[rating_col].apply(label_sentiment)
    df = df.dropna(subset=["sentimen"]).copy()

    # Bersihkan dan batasi agar praktikum lebih ringan.
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 0].copy()
    df = maybe_balance_minimum(df, "sentimen", min_count=250)

    # Plot EDA awal
    manifest = []

    rating_counts = df[rating_col].round().value_counts().sort_index()
    plot_bar(rating_counts, "Distribusi Rating", "Rating", "Jumlah", "eda_distribusi_rating.png")
    add_manifest_item(manifest, "plots/eda_distribusi_rating.png", "Distribusi Rating", "Sebaran rating ulasan.")

    sent_counts = df["sentimen"].value_counts().reindex(["Negatif", "Netral", "Positif"])
    plot_bar(sent_counts, "Distribusi Sentimen", "Sentimen", "Jumlah", "eda_distribusi_sentimen.png")
    add_manifest_item(manifest, "plots/eda_distribusi_sentimen.png", "Distribusi Sentimen", "Sebaran kelas sentimen.")

    df["panjang_karakter"] = df[text_col].str.len()
    plot_histogram(df["panjang_karakter"], "Panjang Review (Karakter)", "Jumlah Karakter", "eda_panjang_karakter.png", bins=40)
    add_manifest_item(manifest, "plots/eda_panjang_karakter.png", "Panjang Review", "Histogram panjang review.")

    df["panjang_kata"] = df[text_col].str.split().apply(len)
    fig = plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="sentimen", y="panjang_kata", order=["Negatif", "Netral", "Positif"])
    plt.title("Panjang Review per Sentimen")
    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah Kata")
    save_plot(fig, "eda_boxplot_panjang_kata.png")
    add_manifest_item(manifest, "plots/eda_boxplot_panjang_kata.png", "Boxplot Panjang Kata", "Perbandingan panjang review antar kelas.")

    # Wordcloud dan top words
    negatif = df[df["sentimen"] == "Negatif"][text_col]
    netral = df[df["sentimen"] == "Netral"][text_col]
    positif = df[df["sentimen"] == "Positif"][text_col]

    plot_wordcloud(positif, "WordCloud Positif", "eda_wordcloud_positif.png")
    add_manifest_item(manifest, "plots/eda_wordcloud_positif.png", "WordCloud Positif", "Kata yang sering muncul pada ulasan positif.")

    plot_wordcloud(negatif, "WordCloud Negatif", "eda_wordcloud_negatif.png")
    add_manifest_item(manifest, "plots/eda_wordcloud_negatif.png", "WordCloud Negatif", "Kata yang sering muncul pada ulasan negatif.")

    plot_top_words(positif, "Top 20 Kata Positif", "eda_top_kata_positif.png")
    add_manifest_item(manifest, "plots/eda_top_kata_positif.png", "Top Kata Positif", "Frekuensi kata pada ulasan positif.")

    plot_top_words(negatif, "Top 20 Kata Negatif", "eda_top_kata_negatif.png")
    add_manifest_item(manifest, "plots/eda_top_kata_negatif.png", "Top Kata Negatif", "Frekuensi kata pada ulasan negatif.")

    # Persiapan label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["sentimen"])
    class_names = list(label_encoder.classes_)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[text_col].tolist())
    sequences = tokenizer.texts_to_sequences(df[text_col].tolist())
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model(num_classes=len(class_names))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc_path, loss_path = plot_history(history)
    add_manifest_item(manifest, "plots/cnn_accuracy.png", "Accuracy CNN", "Grafik akurasi training dan validasi.")
    add_manifest_item(manifest, "plots/cnn_loss.png", "Loss CNN", "Grafik loss training dan validasi.")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, "cnn_confusion_matrix.png")
    add_manifest_item(manifest, "plots/cnn_confusion_matrix.png", "Confusion Matrix", "Matriks kesalahan prediksi model.")

    # ROC multiclass
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    if y_test_bin.ndim == 1:
        y_test_bin = np.column_stack([1 - y_test_bin, y_test_bin])

    fig = plt.figure(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_prob[:, idx])
            auc_score = roc_auc_score(y_test_bin[:, idx], y_prob[:, idx])
            plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_score:.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.title("ROC Curve Multiclass")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_plot(fig, "cnn_roc_multiclass.png")
    add_manifest_item(manifest, "plots/cnn_roc_multiclass.png", "ROC Curve", "Kurva ROC untuk setiap kelas.")

    # Precision-recall sederhana
    fig = plt.figure(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        try:
            precision, recall, _ = precision_recall_curve(y_test_bin[:, idx], y_prob[:, idx])
            plt.plot(recall, precision, label=class_name)
        except Exception:
            continue
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    save_plot(fig, "cnn_precision_recall.png")
    add_manifest_item(manifest, "plots/cnn_precision_recall.png", "Precision-Recall", "Kurva precision-recall multiclass.")

    # Simpan model dan aset
    model.save(MODEL_PATH)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=class_names)

    hasil = {
        "dataset": str(data_file),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "rating_col": rating_col,
        "text_col": text_col,
        "sentimen_counts": {k: int(v) for k, v in df["sentimen"].value_counts().items()},
        "class_names": class_names,
        "accuracy": float(test_acc),
        "loss": float(test_loss),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "classification_report_text": report_text,
        "plots": manifest,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nSelesai training.")
    print("Accuracy:", test_acc)
    print("Loss:", test_loss)
    print("Model:", MODEL_PATH)
    print("Report:", REPORT_PATH)
    print("Plot manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    # Supaya hasil konsisten
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    main()

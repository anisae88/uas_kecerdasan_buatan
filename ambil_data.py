import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PLOTS_DIR = DATA_DIR / "plots"
DEFAULT_DATASET = "farhan999/tokopedia-product-reviews"
OUTPUT_FILE = DATA_DIR / "tokopedia_reviews.csv"


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


def find_csv_file(folder: Path) -> Path | None:
    csvs = list(folder.rglob("*.csv"))
    if not csvs:
        return None
    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return csvs[0]


def download_with_kagglehub(dataset: str, output_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Paket kagglehub belum terpasang. Jalankan: pip install -r requirements.txt"
        ) from exc

    print(f"Mengunduh dataset Kaggle: {dataset}")
    # BAGIAN YANG DIPERBAIKI: Menghapus argumen output_dir
    downloaded_path = kagglehub.dataset_download(dataset)
    downloaded_path = Path(downloaded_path)
    print(f"Folder hasil unduh: {downloaded_path}")
    return downloaded_path


def prepare_local_dataset(source_csv: Path, output_csv: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    safe_mkdir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    return df


def make_eda_plots(df: pd.DataFrame, rating_col: str, text_col: str) -> None:
    safe_mkdir(PLOTS_DIR)

    ratings = pd.to_numeric(df[rating_col], errors="coerce").dropna()
    plt.figure(figsize=(8, 5))
    ratings.value_counts().sort_index().plot(kind="bar")
    plt.title("Distribusi Rating")
    plt.xlabel("Rating")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ambil_data_distribusi_rating.png", dpi=160)
    plt.close()

    lengths = df[text_col].astype(str).str.len()
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=40)
    plt.title("Panjang Teks Review")
    plt.xlabel("Jumlah Karakter")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ambil_data_panjang_teks.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    lengths.describe().plot(kind="bar")
    plt.title("Statistik Panjang Teks")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ambil_data_statistik_panjang_teks.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Praktikum Kaggle Tokopedia Reviews")
    parser.add_argument("--download", action="store_true", help="Unduh dataset dari Kaggle via kagglehub")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Slug dataset Kaggle")
    parser.add_argument("--sample", type=int, default=3000, help="Jumlah maksimum baris untuk file kerja")
    parser.add_argument("--source", default="", help="Path file CSV lokal jika tidak download")
    args = parser.parse_args()

    safe_mkdir(DATA_DIR)
    safe_mkdir(RAW_DIR)
    safe_mkdir(PLOTS_DIR)

    print("=" * 72)
    print("PENGENALAN KAGGLE DAN PEMBACAAN DATA TOKOPEDIA")
    print("=" * 72)
    print("Dataset contoh:", args.dataset)

    source_csv = None
    if args.download:
        downloaded_folder = download_with_kagglehub(args.dataset, RAW_DIR)
        source_csv = find_csv_file(downloaded_folder)
        if source_csv is None:
            raise SystemExit("Tidak menemukan file CSV setelah unduh dataset.")
    else:
        if args.source:
            source_csv = Path(args.source)
        else:
            preferred = DATA_DIR / "tokopedia_reviews_raw.csv"
            if preferred.exists():
                source_csv = preferred
            elif OUTPUT_FILE.exists():
                source_csv = OUTPUT_FILE
            else:
                raise SystemExit(
                    "File CSV tidak ditemukan. Letakkan file di data/ atau gunakan --download."
                )

    print(f"File sumber: {source_csv}")
    raw_copy = DATA_DIR / "tokopedia_reviews_raw.csv"
    shutil.copy2(source_csv, raw_copy)

    df = prepare_local_dataset(raw_copy, OUTPUT_FILE, max_rows=args.sample)

    print("\nRingkasan data:")
    print("Bentuk data:", df.shape)
    print("Kolom:", list(df.columns))

    rating_col = detect_column(df.columns, ["rating", "score", "stars", "bintang", "nilai"])
    text_col = detect_column(df.columns, ["review", "content", "text", "ulasan", "comment", "komentar", "message"])

    if rating_col is None or text_col is None:
        print("\nKolom terdeteksi otomatis:")
        print("rating:", rating_col)
        print("teks:", text_col)
        print("Silakan sesuaikan nama kolom di skrip jika diperlukan.")
    else:
        print(f"\nKolom rating terdeteksi: {rating_col}")
        print(f"Kolom teks terdeteksi: {text_col}")
        make_eda_plots(df, rating_col, text_col)

    print("\n5 data pertama:")
    print(df.head())

    print("\nMissing value:")
    print(df.isnull().sum())

    print(f"\nFile kerja disimpan ke: {OUTPUT_FILE}")
    print("Grafik awal disimpan ke folder data/plots/")
    print("\nSelesai.")


if __name__ == "__main__":
    main()
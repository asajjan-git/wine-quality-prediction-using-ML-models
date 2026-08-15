"""
Downloads the UCI Wine Quality (white wine) dataset into this folder.

Usage:
    python download_data.py
"""
import os
import urllib.request

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winequality-white.csv")


def main():
    if os.path.exists(OUT_PATH):
        print(f"Already have it: {OUT_PATH}")
        return

    print(f"Downloading {URL} ...")
    try:
        urllib.request.urlretrieve(URL, OUT_PATH)
        print(f"Saved to {OUT_PATH}")
    except Exception as e:
        raise SystemExit(
            f"Could not download the dataset automatically ({e}).\n"
            "Download it manually from:\n"
            f"  {URL}\n"
            f"and save it as: {OUT_PATH}\n"
            "Or install the 'ucimlrepo' package and use fetch_ucirepo(id=186)."
        )


if __name__ == "__main__":
    main()

"""
Download MovieLens 20M dataset from Kaggle.
Assumes Kaggle API is configured (~/.kaggle/kaggle.json).
"""

import os
import subprocess
import zipfile
import shutil
from pathlib import Path

base = Path(__file__).resolve().parents[1]
raw_dir = base / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

required_files = [
    "ratings.csv",
    "tags.csv",
    "movies.csv",
    "links.csv",
    "genome_tags.csv",
    "genome_scores.csv",
]

# Skip download if files already exist
if all((raw_dir / f).exists() for f in required_files):
    print("[INFO] All required files already present. Skipping download.")
else:
    print("[INFO] Downloading MovieLens 20M dataset...")
    try:
        os.chdir(raw_dir)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "grouplens/movielens-20m-dataset"],
            check=True,
        )

        zip_file = raw_dir / "movielens-20m-dataset.zip"
        if not zip_file.exists():
            raise FileNotFoundError("Download did not produce the expected zip file.")

        print("[INFO] Extracting dataset...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

        extracted_dir = raw_dir / "ml-20m"
        if extracted_dir.exists():
            for file in required_files:
                src, dst = extracted_dir / file, raw_dir / file
                if src.exists():
                    src.rename(dst)
                    print(f"[INFO] Extracted {file}")
                else:
                    print(f"[WARN] {file} not found in extracted folder.")

            shutil.rmtree(extracted_dir, ignore_errors=True)
            zip_file.unlink()

        print("[INFO] Download complete! Files are ready in data/raw/")

    except subprocess.CalledProcessError:
        print("[ERROR] Kaggle API command failed. Ensure Kaggle is installed and configured.")
    except Exception as e:
        print(f"[ERROR] Unexpected issue: {e}")

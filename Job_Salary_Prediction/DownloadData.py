import kaggle
from pathlib import Path
import shutil

# Download dataset
dataset = "ruchi798/data-science-job-salaries"
kaggle.api.dataset_download_files(dataset, path="Day4_JobSalaryPrediction/data", unzip=True)

# Rename to jobs.csv
data_dir = Path("Day4_JobSalaryPrediction/data")
csv_file = data_dir / "ds_salaries.csv"
shutil.move(csv_file, data_dir / "jobs.csv")
print("Dataset downloaded and renamed to jobs.csv")
import os
import random
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.const import DATA_DIRECTORY, MODELS_DIRECTORY

if not os.path.exists(DATA_DIRECTORY):
    DATA_DIRECTORY = "data"
if not os.path.exists(MODELS_DIRECTORY):
    MODELS_DIRECTORY = "models"


def generate_subset_dataframe(dataset, subset_fraction=0.7):
    x_col = "filename"
    y_col = "class"
    columns = [x_col, y_col]
    subset = pd.DataFrame(columns=columns)

    for subdir, dirs, files in os.walk(os.path.join(DATA_DIRECTORY, dataset, "train")):
        for file in files:
            if random.random() <= subset_fraction:
                src = Path(os.path.join(subdir, file))
                src = Path(*src.parts[1:])
                y = f"{os.path.basename(os.path.dirname(src))}_class"
                subset = subset.append({x_col: src, y_col: y}, ignore_index=True)

    filename = f'subset_{str(subset_fraction).replace(".", "")}_{uuid.uuid4().hex}.csv'
    save_path = os.path.join(MODELS_DIRECTORY, dataset, "subsets")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    subset.to_csv(os.path.join(save_path, filename), index=False)


if __name__ == "__main__":
    for i in range(int(sys.argv[1])):
        generate_subset_dataframe(sys.argv[2], subset_fraction=float(sys.argv[3]))

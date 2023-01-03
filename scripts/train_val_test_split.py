import os
import random
import shutil
import sys
from pathlib import Path

from config.const import DATA_DIRECTORY, MODELS_DIRECTORY

if not os.path.exists(DATA_DIRECTORY):
    DATA_DIRECTORY = "data"
if not os.path.exists(MODELS_DIRECTORY):
    MODELS_DIRECTORY = "models"


def replace_path(path, frm, to):
    pre, match, post = path.rpartition(frm)
    return "".join((to if match else pre, match, post))


def create_val(dataset, validation_split=0.2):
    train_path = os.path.join(DATA_DIRECTORY, dataset, "train")
    val_path = os.path.join(DATA_DIRECTORY, dataset, "val")

    if not os.path.exists(train_path):
        print("train directory does not exist")
        return None

    if os.path.exists(val_path):
        print("val directory already exist")
        return None

    print("create val directory")

    for subdir, dirs, files in os.walk(os.path.join(DATA_DIRECTORY, dataset, "train")):
        for file in files:
            if random.random() <= validation_split:

                src = os.path.join(subdir, file)
                dst = Path(src)
                index = dst.parts.index("train")
                dst = Path(val_path).joinpath(*dst.parts[index + 1 :])

                print(f"src:{src}, dst:{dst}")
                Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)


if __name__ == "__main__":
    create_val(str(sys.argv[1]))

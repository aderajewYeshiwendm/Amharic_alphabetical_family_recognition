import random
import shutil
from pathlib import Path

# -------- CONFIG --------
SRC_DIR = Path("dataset")          # source folder
OUT_DIR = Path("dataset_split")    # output folder

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # for reproducibility
# ------------------------

classes = [d for d in SRC_DIR.iterdir() if d.is_dir()]

for split in ["train", "val", "test"]:
    for cls in classes:
        (OUT_DIR / split / cls.name).mkdir(parents=True, exist_ok=True)

for cls in classes:
    files = list(cls.iterdir())
    files = [f for f in files if f.is_file()]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    for f in train_files:
        shutil.copy2(f, OUT_DIR / "train" / cls.name / f.name)

    for f in val_files:
        shutil.copy2(f, OUT_DIR / "val" / cls.name / f.name)

    for f in test_files:
        shutil.copy2(f, OUT_DIR / "test" / cls.name / f.name)

    print(f"{cls.name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("\n✅ Dataset split completed.")

import random
import shutil
from pathlib import Path
from collections import defaultdict

SRC_DIR = Path("dataset")         
OUT_DIR = Path("dataset_split")    
AUG_SUFFIXES = ["bold", "orig", "rot", "zoom", "shift", "thin"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

random.seed(42)  
classes = [d for d in SRC_DIR.iterdir() if d.is_dir()]

for split in ["train", "val", "test"]:
    for cls in classes:
        (OUT_DIR / split / cls.name).mkdir(parents=True, exist_ok=True)

for cls in classes:
    all_files = [f for f in cls.iterdir() if f.is_file()]
    
    
    groups = defaultdict(list)
    for f in all_files:
        name_stem = f.stem 
        root_name = name_stem
        
    
        for sfx in AUG_SUFFIXES:
            if name_stem.endswith(f"_{sfx}"):
                root_name = name_stem.rsplit(f"_{sfx}", 1)[0]
                break
        
        groups[root_name].append(f)
    
    unique_roots = list(groups.keys())
    random.shuffle(unique_roots)

    n = len(unique_roots)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_roots = unique_roots[:n_train]
    val_roots = unique_roots[n_train:n_train + n_val]
    test_roots = unique_roots[n_train + n_val:]

    
    def copy_group_to_split(roots, split_label):
        file_count = 0
        for root in roots:
            for f in groups[root]:
                shutil.copy2(f, OUT_DIR / split_label / cls.name / f.name)
                file_count += 1
        return file_count

    tr_count = copy_group_to_split(train_roots, "train")
    va_count = copy_group_to_split(val_roots, "val")
    te_count = copy_group_to_split(test_roots, "test")

    print(f"Class [{cls.name}]:")
    print(f"  - {len(train_roots)} original groups -> {tr_count} total images in TRAIN")
    print(f"  - {len(val_roots)} original groups -> {va_count} total images in VAL")
    print(f"  - {len(test_roots)} original groups -> {te_count} total images in TEST")
    print("-" * 30)

print("\n✅ split completed")

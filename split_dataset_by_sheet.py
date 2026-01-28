"""
Split dataset by SHEET ID to prevent data leakage.

This script ensures that all augmented versions of characters from the same 
scanned sheet stay together in either train, val, or test set.

Filename format: {sample_id}_{char_id}_{unique_id}_{type_name}.png
Example: dataset0_form1_7956_bold.png
         ^^^^^^^^^
         sheet ID (sample_id)
"""

import random
import shutil
from pathlib import Path
from collections import defaultdict

# -------- CONFIG --------
SRC_DIR = Path("dataset")          # source folder with current data
OUT_DIR = Path("dataset_sheet_split")    # output folder for new split

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # for reproducibility
# ------------------------

def extract_sheet_id(filename):
    """
    Extract sheet ID from filename.
    Example: 'dataset0_form1_7956_bold.png' -> 'dataset0'
    """
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0]
    return None

def group_images_by_sheet(src_dir):
    """
    Group all images by their sheet ID.
    Handles both flat structure (class folders) and nested structure (train/val/test/class).
    Returns: dict[class_name][sheet_id] = [list of file paths]
    """
    class_sheet_groups = {}
    
    # Check if dataset is already split (has train/val/test folders)
    has_splits = any(d.name in ['train', 'val', 'test'] for d in src_dir.iterdir() if d.is_dir())
    
    if has_splits:
        print("📂 Detected pre-split dataset (train/val/test folders)")
        print("   Merging all splits to re-split by sheet ID...")
        
        # Collect all images from train/val/test
        all_classes = set()
        for split_dir in src_dir.iterdir():
            if split_dir.is_dir() and split_dir.name in ['train', 'val', 'test']:
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        all_classes.add(class_dir.name)
        
        # Group by class and sheet
        for class_name in sorted(all_classes):
            sheet_groups = defaultdict(list)
            
            # Collect from all splits
            for split_name in ['train', 'val', 'test']:
                split_class_dir = src_dir / split_name / class_name
                if split_class_dir.exists():
                    for img_file in split_class_dir.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                            sheet_id = extract_sheet_id(img_file.name)
                            if sheet_id:
                                sheet_groups[sheet_id].append(img_file)
                            else:
                                print(f"⚠️  Warning: Could not extract sheet ID from {img_file.name}")
            
            class_sheet_groups[class_name] = sheet_groups
            total_images = sum(len(files) for files in sheet_groups.values())
            print(f"✓ {class_name}: Found {len(sheet_groups)} unique sheets, {total_images} images")
    else:
        print("📂 Detected flat dataset structure (class folders only)")
        
        # Iterate through each class folder
        for class_dir in sorted(src_dir.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            sheet_groups = defaultdict(list)
            
            # Group files by sheet ID
            for img_file in class_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    sheet_id = extract_sheet_id(img_file.name)
                    if sheet_id:
                        sheet_groups[sheet_id].append(img_file)
                    else:
                        print(f"⚠️  Warning: Could not extract sheet ID from {img_file.name}")
            
            class_sheet_groups[class_name] = sheet_groups
            total_images = sum(len(files) for files in sheet_groups.values())
            print(f"✓ {class_name}: Found {len(sheet_groups)} unique sheets, {total_images} images")
    
    return class_sheet_groups

def split_sheets(sheet_ids, train_ratio, val_ratio):
    """
    Split sheet IDs into train/val/test sets.
    Returns: (train_sheets, val_sheets, test_sheets)
    """
    sheet_list = list(sheet_ids)
    random.shuffle(sheet_list)
    
    n = len(sheet_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_sheets = sheet_list[:n_train]
    val_sheets = sheet_list[n_train:n_train + n_val]
    test_sheets = sheet_list[n_train + n_val:]
    
    return train_sheets, val_sheets, test_sheets

def copy_files_by_sheet(class_sheet_groups, out_dir, train_sheets, val_sheets, test_sheets):
    """
    Copy files to train/val/test directories based on sheet assignment.
    """
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }
    
    for class_name, sheet_groups in class_sheet_groups.items():
        # Create output directories
        for split in ['train', 'val', 'test']:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy files based on sheet assignment
        for sheet_id, files in sheet_groups.items():
            if sheet_id in train_sheets:
                split = 'train'
            elif sheet_id in val_sheets:
                split = 'val'
            elif sheet_id in test_sheets:
                split = 'test'
            else:
                print(f"⚠️  Warning: Sheet {sheet_id} not assigned to any split")
                continue
            
            # Copy all files from this sheet to the assigned split
            for file_path in files:
                dest = out_dir / split / class_name / file_path.name
                shutil.copy2(file_path, dest)
                stats[split][class_name] += 1
    
    return stats

def print_statistics(stats, class_sheet_groups):
    """
    Print dataset statistics after splitting.
    """
    print("\n" + "="*60)
    print("DATASET SPLIT STATISTICS (by sheet)")
    print("="*60)
    
    all_classes = sorted(stats['train'].keys())
    
    # Per-class statistics
    print("\nPer-class distribution:")
    print(f"{'Class':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_name in all_classes:
        train_count = stats['train'][class_name]
        val_count = stats['val'][class_name]
        test_count = stats['test'][class_name]
        total = train_count + val_count + test_count
        
        print(f"{class_name:<10} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
    
    print("-" * 60)
    total_all = total_train + total_val + total_test
    print(f"{'TOTAL':<10} {total_train:<10} {total_val:<10} {total_test:<10} {total_all:<10}")
    
    # Percentage distribution
    print(f"\nPercentage distribution:")
    print(f"  Train: {total_train}/{total_all} ({100*total_train/total_all:.1f}%)")
    print(f"  Val:   {total_val}/{total_all} ({100*total_val/total_all:.1f}%)")
    print(f"  Test:  {total_test}/{total_all} ({100*total_test/total_all:.1f}%)")
    
    # Sheet statistics
    print(f"\nSheet distribution:")
    # Count unique sheets per class
    for class_name in all_classes:
        num_sheets = len(class_sheet_groups[class_name])
        print(f"  {class_name}: {num_sheets} unique sheets")

def main():
    """
    Main function to split dataset by sheet ID.
    """
    if not SRC_DIR.exists():
        print(f"❌ Error: Source directory not found: {SRC_DIR}")
        return
    
    print("="*60)
    print("SPLITTING DATASET BY SHEET ID")
    print("="*60)
    print(f"\nSource: {SRC_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print()
    
    # Step 1: Group images by sheet
    print("Step 1: Grouping images by sheet ID...")
    class_sheet_groups = group_images_by_sheet(SRC_DIR)
    
    # Step 2: Get all unique sheet IDs (across all classes)
    print("\nStep 2: Finding unique sheet IDs...")
    all_sheet_ids = set()
    for class_name, sheet_groups in class_sheet_groups.items():
        all_sheet_ids.update(sheet_groups.keys())
    
    print(f"✓ Found {len(all_sheet_ids)} unique sheets total")
    print(f"  Sheet IDs: {sorted(all_sheet_ids)[:10]}{'...' if len(all_sheet_ids) > 10 else ''}")
    
    if len(all_sheet_ids) == 0:
        print("\n❌ Error: No sheet IDs found! Check your dataset structure and filename format.")
        print("   Expected filename format: {sample_id}_{char_id}_{unique_id}_{type_name}.png")
        print("   Example: dataset0_form1_7956_bold.png")
        return
    
    # Step 3: Split sheets into train/val/test
    print("\nStep 3: Splitting sheets into train/val/test...")
    train_sheets, val_sheets, test_sheets = split_sheets(
        all_sheet_ids, TRAIN_RATIO, VAL_RATIO
    )
    
    print(f"✓ Train sheets: {len(train_sheets)} ({100*len(train_sheets)/len(all_sheet_ids):.1f}%)")
    print(f"✓ Val sheets:   {len(val_sheets)} ({100*len(val_sheets)/len(all_sheet_ids):.1f}%)")
    print(f"✓ Test sheets:  {len(test_sheets)} ({100*len(test_sheets)/len(all_sheet_ids):.1f}%)")
    
    # Convert to sets for fast lookup
    train_sheets = set(train_sheets)
    val_sheets = set(val_sheets)
    test_sheets = set(test_sheets)
    
    # Step 4: Copy files based on sheet assignment
    print("\nStep 4: Copying files to train/val/test directories...")
    stats = copy_files_by_sheet(
        class_sheet_groups, OUT_DIR, train_sheets, val_sheets, test_sheets
    )
    
    # Step 5: Print statistics
    print_statistics(stats, class_sheet_groups)
    
    print("\n" + "="*60)
    print("✅ DATASET SPLIT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nNew dataset location: {OUT_DIR.absolute()}")
    print("\nNext steps:")
    print("  1. Update your training notebook to use the new dataset path:")
    print(f"     data_dir = '{OUT_DIR}/'")
    print("  2. Retrain your model with the proper split")
    print("  3. Compare the new accuracy (expected to be lower, but more realistic)")

if __name__ == "__main__":
    main()

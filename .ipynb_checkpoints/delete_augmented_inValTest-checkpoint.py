import os
from pathlib import Path

# -------- CONFIG --------
BASE_DIR = Path("dataset_split")  # The folder containing your train, val, test splits
FOLDERS_TO_CLEAN = ["val", "test"] # We only clean these two
REQUIRED_KEYWORD = "orig"         # We only keep files with this string
# ------------------------

def clean_augmented_files(base_path, split_folders):
    deleted_count = 0
    kept_count = 0
    
    for split in split_folders:
        split_path = base_path / split
        
        if not split_path.exists():
            print(f"⚠️ Warning: {split_path} does not exist. Skipping.")
            continue
            
        # Walk through all family subfolders (e.g., Ha_family, Be_family)
        for root, dirs, files in os.walk(split_path):
            for filename in files:
                file_path = Path(root) / filename
                
                # Check if 'orig' is NOT in the name
                if REQUIRED_KEYWORD not in filename.lower():
                    try:
                        file_path.unlink() # Delete the file
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ Error deleting {filename}: {e}")
                else:
                    kept_count += 1
                    
    print(f"\n✅ Cleanup Finished for {split_folders}:")
    print(f"🗑️ Deleted {deleted_count} augmented images.")
    print(f"📦 Kept {kept_count} original images.")

# Execute the cleanup
clean_augmented_files(BASE_DIR, FOLDERS_TO_CLEAN)

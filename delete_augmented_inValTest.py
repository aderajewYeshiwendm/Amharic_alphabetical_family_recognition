import os
from pathlib import Path

BASE_DIR = Path("dataset_split")
FOLDERS_TO_CLEAN = ["val", "test"]
REQUIRED_KEYWORD = "orig" 

def clean_augmented_files(base_path, split_folders):
    deleted_count = 0
    kept_count = 0
    
    for split in split_folders:
        split_path = base_path / split
        
        if not split_path.exists():
            print(f"⚠️ Warning: {split_path} does not exist. Skipping.")
            continue
            
        for root, dirs, files in os.walk(split_path):
            for filename in files:
                file_path = Path(root) / filename
                
                if REQUIRED_KEYWORD not in filename.lower():
                    try:
                        file_path.unlink() 
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ Error deleting {filename}: {e}")
                else:
                    kept_count += 1
                    
    print(f"\n✅ Cleanup Finished for {split_folders}:")
    print(f"🗑️ Deleted {deleted_count} augmented images.")
    print(f"📦 Kept {kept_count} original images.")

clean_augmented_files(BASE_DIR, FOLDERS_TO_CLEAN)

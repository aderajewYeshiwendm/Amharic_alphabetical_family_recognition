# Amharic Alphabet Family Recognition Using Deep Learning

## 📌 Project Overview
This project focuses on recognizing **Amharic alphabetical family characters (ሀ to በ)** using deep learning techniques.  
The goal is to build an end-to-end deep learning system starting from **custom data collection** to **model design, training, and evaluation**.

The system processes scanned handwritten worksheets containing a 10×7 grid of characters, automatically extracts individual character images, applies data augmentation, and trains a Convolutional Neural Network (CNN) to classify characters into 10 alphabet families. The trained model achieves **92% accuracy** on truly unseen test sheets, demonstrating strong generalization to new handwriting samples.

---

## 🎯 Objectives
- Collect a **custom dataset** of Amharic alphabetical family characters (ሀ to በ)
- Extract characters from scanned worksheets using perspective transformation
- Preprocess and augment the dataset (6 variations per character)
- Design and train a **PyTorch CNN model** for character recognition
- Evaluate model performance using accuracy, confusion matrix, and loss curves

---

## 🔠 Alphabet Classes
The project focuses on the following **10 Amharic alphabet family characters**:

- **ሀ** (ha)
- **ለ** (le)
- **ሐ** (hha)
- **መ** (me)
- **ሠ** (se)
- **ረ** (re)
- **ሰ** (sa)
- **ሸ** (sha)
- **ቀ** (qe)
- **በ** (be)

---

## 📊 Dataset Collection
- The dataset was **collected manually** by scanning handwritten worksheets
- **80 scanned sheets** (JPG images) containing 10×7 grids of characters
- Each sheet contains 70 character cells (10 rows × 7 columns)
- Characters were written by multiple individuals to ensure diversity
- Images were captured using phone cameras and stored in `scanned_sheets/` directory
- No open-source or publicly available datasets were used

**Dataset Statistics:**
- Original scanned sheets: **80 JPG files**
- **72 unique sheets** successfully processed (some skipped due to calibration issues)
- Extracted character images (after augmentation): **22,402 PNG images**

**Proper Dataset Split (by sheet ID to prevent data leakage):**
- **Training set**: 51 sheets → **15,807 images** (70.6%)
- **Validation set**: 10 sheets → **2,710 images** (12.1%)
- **Test set**: 12 sheets → **3,885 images** (17.3%)

**Key Feature**: Dataset is split by **sheet ID**, ensuring all augmented versions of characters from the same sheet stay together in one split. This prevents data leakage and provides realistic evaluation of model generalization.

---

## 🧹 Data Preprocessing

### Grid Extraction (`grid_cell_extractor.ipynb`)
1. **Manual Perspective Calibration**: Interactive 4-corner clicking to correct perspective distortion
2. **Perspective Transformation**: Warp the scanned image to a rectangular grid
3. **Grid Slicing**: Extract individual character cells from the 10×7 grid
4. **Image Processing**:
   - Grayscale conversion
   - Bilateral filtering for noise reduction
   - Otsu thresholding for binarization
   - Contour detection and bounding box extraction
   - Resize to 64×64 pixels with padding

### Data Augmentation
Each extracted character generates **6 variations**:
- **Original**: Unmodified character image
- **Rotation**: Random rotation between ±5° to ±10°
- **Zoom**: Random zoom factor between 1.1× to 1.2×
- **Shift**: Random translation up to ±3 pixels
- **Bold**: Morphological dilation (thickening)
- **Thin**: Morphological erosion (thinning) or fallback rotation

### Dataset Splitting (`split_dataset_by_sheet.py`)
**Proper Split Method** (prevents data leakage):
- Splits by **sheet ID**, not individual images
- All augmented versions of characters from the same sheet stay together
- Ensures model is evaluated on truly unseen handwriting samples

**Split Ratios:**
- Training: 70% of sheets (51 sheets)
- Validation: 15% of sheets (10 sheets)
- Test: 15% of sheets (12 sheets)

**Why this matters:**
- Original `split_dataset.py` split randomly at image level (causing data leakage)
- Augmented versions of same character could appear in both train and test
- This inflated accuracy from 89% to 98%
- New method provides **honest evaluation** of generalization

**Directory Structure:**
- Images organized in `dataset_sheet_split/train/`, `val/`, `test/`
- Each split contains subfolders for the 10 alphabet families

---

## 🧠 Model Architecture

### CNN Architecture (`family_recognition_model.ipynb`)
The model uses a **PyTorch-based Convolutional Neural Network** with the following structure:

**Feature Extraction Layers:**
- Conv2d(3, 32, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
- Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
- Conv2d(64, 128, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)

**Classification Layers:**
- Flatten → Linear(128×16×16, 256) → ReLU → Dropout(0.5)
- Linear(256, 10) → Softmax (for 10 classes)

**Training Configuration:**
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: CrossEntropyLoss
- Batch Size: 16
- Epochs: 10 (with early stopping, patience=5)
- Input Size: 128×128×3 (grayscale converted to 3-channel)

The model is trained to classify input images into one of the 10 Amharic alphabet family classes.

---

## ⚙️ Tools & Technologies
- **Python 3**
- **PyTorch** - Deep learning framework
- **OpenCV** - Image processing and computer vision
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization and interactive calibration
- **scikit-learn** - Metrics and evaluation
- **Pillow (PIL)** - Image loading and manipulation
- **seaborn** - Confusion matrix visualization
- **Jupyter Notebook** - Interactive development environment

---

## 📈 Evaluation Metrics

### Overall Performance
- **Test Accuracy**: **92%** on truly unseen sheets
- **Training Accuracy**: 98.06% (9 epochs with early stopping)
- **Validation Accuracy**: 90.96% (best at epoch 6)
- **Training Strategy**: Early stopping with patience=5 to prevent overfitting

### Per-Class Performance (Precision/Recall/F1-Score)

**Excellent Performers (≥97% precision):**
- **qe** (ቀ): Precision 99%, Recall 89%, F1 94% - Highest precision
- **re** (ረ): Precision 97%, Recall 97%, F1 97% - Best balanced performer
- **be** (በ): Precision 97%, Recall 97%, F1 97% - Excellent all-round
- **sha** (ሸ): Precision 97%, Recall 93%, F1 95% - Very strong

**Strong Performers (90-96%):**
- **ha** (ሀ): Precision 96%, Recall 95%, F1 96% - Highly consistent
- **hha** (ሐ): Precision 90%, Recall 92%, F1 91% - Well balanced

**Good Performers (85-89%):**
- **se** (ሠ): Precision 88%, Recall 95%, F1 91% - Good recall
- **me** (መ): Precision 87%, Recall 91%, F1 89% - Solid performance
- **le** (ለ): Precision 85%, Recall 86%, F1 85% - Consistent

**Most Challenging Class (83%):**
- **sa** (ሰ): Precision 83%, Recall 83%, F1 83% - Requires improvement

### Visualization
- **Loss Curves**: Training and validation loss over epochs
- **Accuracy Curves**: Training and validation accuracy progression
- **Confusion Matrix**: Detailed per-class confusion analysis
- **Classification Report**: Comprehensive precision, recall, F1 metrics

---

## 📁 Project Structure

```text
Amharic_alphabetical_family_recognition/
|
+-- scanned_sheets/          # Original scanned worksheet images (80 JPG files)
|   +-- dataset0.jpg
|   +-- dataset1.jpg
|   +-- ...
|   +-- dataset80.jpg
|   +-- calibration.txt
|
+-- dataset/                 # Initial extracted images (22,566 PNG files)
|   +-- train/, val/, test/ # Original split (has data leakage - not recommended)
|
+-- dataset_sheet_split/
|   +-- train/              # Training set: 51 sheets (15,971 images)
|   |   +-- ha/, le/, hha/, me/, se/, re/, sa/, sha/, qe/, be/
|   +-- val/                # Validation set: 10 sheets (2,710 images)
|   |   +-- ha/, le/, hha/, me/, se/, re/, sa/, sha/, qe/, be/
|   +-- test/               # Test set: 12 sheets (3,885 images)
|       +-- ha/, le/, hha/, me/, se/, re/, sa/, sha/, qe/, be/
|
+-- grid_cell_extractor.ipynb         # Data extraction and augmentation notebook
+-- family_recognition_model.ipynb    # Model training and evaluation notebook
+-- split_dataset.py                  # OLD: Image-level split (causes data leakage)
+-- split_dataset_by_sheet.py         # ✅ NEW: Proper sheet-level split
+-- best_model.pth                    # Trained model weights (saved checkpoint)
+-- requirements.txt                  # Python dependencies
+-- README.md                         # Project documentation
+-- PROJECT_ANALYSIS.md               # Detailed project analysis and improvements
+-- CRITICAL_FIX_1_DATA_LEAKAGE.md    # Documentation of data leakage fix
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Amharic_alphabetical_family_recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Step 1: Extract Characters from Scanned Sheets
1. Open `grid_cell_extractor.ipynb` in Jupyter
2. Run all cells to start the extraction process
3. For each scanned sheet, a window will pop up:
   - Click the **4 corners** of the grid in order: Top-Left → Top-Right → Bottom-Right → Bottom-Left
   - Close the window after clicking 4 times
4. The notebook will automatically:
   - Extract all 70 cells from each sheet
   - Apply 6 augmentations per character
   - Save images to `dataset/` organized by family folders

#### Step 2: Split Dataset by Sheet ID (REQUIRED for proper evaluation)
**Important**: This step prevents data leakage and ensures realistic accuracy evaluation.

```bash
python3 split_dataset_by_sheet.py
```

This script will:
- Merge images from the initial `dataset/` folder
- Group images by their sheet ID (extracted from filename)
- Split sheets (not individual images) into train/val/test
- Create `dataset_sheet_split/` with properly separated data
- Display detailed statistics about the split

**Output:**
```
✓ Found 73 unique sheets
✓ Train: 51 sheets (15,971 images)
✓ Val: 10 sheets (2,710 images)
✓ Test: 12 sheets (3,885 images)
```

#### Step 3: Train the Model
1. Open `family_recognition_model.ipynb` in Jupyter
2. **Update the data directory** (line 52):
   ```python
   data_dir = 'dataset_sheet_split/'  # Use the properly split dataset
   ```
3. Run all cells to:
   - Load and preprocess the dataset
   - Define and initialize the CNN model
   - Train the model with early stopping
   - Evaluate on test set and generate metrics
   - Save the best model to `best_model.pth`

**Expected Results:**
- Training accuracy: ~98%
- Validation accuracy: ~89-90%
- Test accuracy: ~89% (realistic, on unseen sheets)

#### Step 4: Make Predictions
Use the `predict_image()` function in the model notebook to classify new character images:
```python
predict_image(model, 'path/to/character/image.png')
```
# Amharic Alphabet Family Recognition Using Deep Learning

## 📌 Project Overview
This project focuses on recognizing **Amharic alphabetical family characters (ሀ to በ)** using deep learning techniques.  
The goal is to build an end-to-end deep learning system starting from **custom data collection** to **model design, training, and evaluation**.

The system processes scanned handwritten worksheets containing a 10×7 grid of characters, automatically extracts individual character images, applies data augmentation, and trains a Convolutional Neural Network (CNN) to classify characters into 10 alphabet families. The trained model achieves **~98% accuracy** on the test set.

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
- Original scanned sheets: **80 images**
- Extracted character images (after augmentation): **~22,500 images**
- Training set: **~15,800 images** (70%)
- Validation set: **~3,300 images** (15%)
- Test set: **~3,400 images** (15%)

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

### Dataset Splitting (`split_dataset.py`)
- Random split at the image level (70% train, 15% validation, 15% test)
- Images organized by family folders: `dataset/train/`, `dataset/val/`, `dataset/test/`
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
- **Accuracy**: ~98% on test set
- **Loss**: Training and validation loss curves
- **Confusion Matrix**: Per-class classification performance
- **Classification Report**: Precision, recall, and F1-score per class
- **Training vs Validation curves**: Monitor overfitting

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
+-- dataset/                 # Processed character images (~22,500 PNG files)
|   +-- train/              # Training set (~15,800 images)
|   |   +-- ha/
|   |   +-- le/
|   |   +-- hha/
|   |   +-- ... (10 family folders)
|   +-- val/                # Validation set (~3,300 images)
|   |   +-- ha/
|   |   +-- ... (10 family folders)
|   +-- test/               # Test set (~3,400 images)
|       +-- ha/
|       +-- ... (10 family folders)
|
+-- grid_cell_extractor.ipynb    # Data extraction and augmentation notebook
+-- family_recognition_model.ipynb # Model training and evaluation notebook
+-- split_dataset.py              # Utility script for dataset splitting
+-- best_model.pth                # Trained model weights (saved checkpoint)
+-- requirements.txt              # Python dependencies
+-- README.md                     # Project documentation
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

#### Step 2: Split Dataset (Optional)
If you need to reorganize the dataset into train/val/test splits:
```bash
python split_dataset.py
```
This will create `dataset_split/` with train, val, and test folders.

#### Step 3: Train the Model
1. Open `family_recognition_model.ipynb` in Jupyter
2. Ensure `dataset/` contains `train/`, `val/`, and `test/` folders
3. Run all cells to:
   - Load and preprocess the dataset
   - Define and initialize the CNN model
   - Train the model with early stopping
   - Evaluate on test set and generate metrics
   - Save the best model to `best_model.pth`

#### Step 4: Make Predictions
Use the `predict_image()` function in the model notebook to classify new character images:
```python
predict_image(model, 'path/to/character/image.png')
```

---

## 📝 Notes
- The current dataset split is done at the **image level**, meaning augmented versions of the same original character may appear in both training and test sets.
- The model achieves high accuracy (~98%) due to the constrained problem (clean, centered characters from structured grids) and data augmentation strategy.
- Manual calibration is required for each scanned sheet to correct perspective distortion.

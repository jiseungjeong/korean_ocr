# Quick Start Guide

## Prerequisites

- Python 3.10 with conda environment (`ml-env`)
- Required packages: numpy, scikit-learn, scikit-image, opencv-python, tqdm, pandas, matplotlib, seaborn

## Project Structure

```
pdp_ocr/
├── archive/
│   └── Hangul Database/
│       └── Hangul Database/        # 64 classes, 100 images each
├── features/
│   └── hog/                        # Extracted HOG features (.npy files)
├── results/
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── worst_classes_samples.png
│   ├── confused_pairs.csv
│   ├── per_class_accuracy.csv
│   └── error_cases/                # Error visualization images
├── dataloader.py
├── feature_extractor.py
├── analysis.py
├── error_case_analysis.py
├── main.ipynb
└── README.md
```

## Step-by-Step Execution

### 1. Extract HOG Features (One-time setup)

```bash
conda activate ml-env
cd /path/to/pdp_ocr
python feature_extractor.py
```

This will:
- Process 6,400 images (64 classes × 100 samples)
- Extract HOG features (orientations=9, pixels_per_cell=(8,8))
- Save to `features/hog/*.npy` (one file per class)
- Takes ~5-10 minutes

### 2. Run Error Analysis

```bash
conda activate ml-env
python analysis.py
```

This generates:
- Confusion matrix (64×64)
- Per-class accuracy metrics
- List of most confused character pairs
- Performance visualizations

**Output**:
- `results/confusion_matrix.png`
- `results/per_class_accuracy.png`
- `results/confused_pairs.csv`
- `results/per_class_accuracy.csv`

### 3. Visualize Error Cases

```bash
conda activate ml-env
python error_case_analysis.py
```

This creates:
- Side-by-side comparison of confused characters
- HOG feature visualizations
- Examples from worst-performing classes

**Output**:
- `results/error_cases/*.png` (10 confusion pairs)
- `results/worst_classes_samples.png`

### 4. View Analysis Report

Open `ANALYSIS_REPORT.md` for comprehensive findings including:
- Performance metrics
- Top confused character pairs
- Structural insights on why errors occur
- Recommendations for improvement

## Main Results

### Current Performance
- **Test Accuracy**: 78.83%
- **Training samples**: 5,120
- **Test samples**: 1,280
- **Feature dimension**: 1,764 (HOG) → 256 (PCA)

### Best Performing Classes (100% accuracy)
- eu (으)
- iss (있)
- won (원)

### Worst Performing Classes
- ji (지): 35%
- si (시): 40%
- in (인): 55%

### Most Confused Pairs
- so (소) ↔ seu (스)
- si (시) ↔ seo (서) ↔ sa (사)
- deul (들) ↔ reul (를)

## Next Steps

See `TODOLIST.md` for additional experiments:
- Hybrid model (CNN features + ML classifier)
- Data augmentation study
- Feature fusion experiments

## Troubleshooting

### Missing packages
```bash
conda activate ml-env
pip install scikit-image opencv-python tqdm
```

### Features not found
Make sure `feature_extractor.py` completed successfully.
Check that `features/hog/` contains 64 `.npy` files.

### Out of memory
Reduce `n_components` in PCA or process fewer samples with `max_samples_per_class`.

## Contact

Team: Group 14  
Course: CSE36301 Machine Learning, UNIST  
Semester: 2025 Fall


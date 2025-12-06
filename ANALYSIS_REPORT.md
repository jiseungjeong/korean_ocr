# Korean Character Recognition - Analysis Report

## Executive Summary

This report presents a comprehensive error analysis of our traditional ML-based Korean character recognition system using HOG features and K-Nearest Neighbors classification with **Extended Dataset** (data augmentation applied).

**Key Findings**:
- Test Accuracy: **84.67%** (4,120 errors out of 26,880 test samples)
- **+5.84%p improvement** over Basic dataset (78.83% → 84.67%)
- PCA preserved **87.39%** of variance with 256 components
- Significant performance improvement on previously difficult classes
- Clear patterns in misclassification related to Korean character structure

---

## Model Performance

### Overall Metrics
- **Dataset**: Extended (with 20x augmentation via Albumentations)
- **Training samples**: 107,520 (21x increase from Basic)
- **Test samples**: 26,880 (21x increase from Basic)
- **Number of classes**: 64
- **Feature dimension**: 1,764 (HOG) → 256 (PCA)
- **Classifier**: K-Nearest Neighbors (k=7)
- **Test Accuracy**: 84.67% (+5.84%p improvement)

### Performance Distribution
- **Excellent performance** (>95%): 3 classes (geos: 98.8%, bak: 96.4%, su: 96.0%)
- **High performance** (90-95%): 7 classes
- **Good performance** (80-90%): 35 classes
- **Moderate performance** (70-80%): 14 classes
- **Challenging performance** (<70%): 5 classes (i: 61%, jeong: 62%, yeo: 64%, ji: 64%, eo: 70%)

---

## Error Analysis

### Top 10 Most Confused Character Pairs

| Rank | True Class | Predicted Class | Error Count | Error Rate |
|------|------------|-----------------|-------------|------------|
| 1 | yeo (여) | eo (어) | 87 | 20.7% |
| 2 | i (이) | eo (어) | 68 | 16.2% |
| 3 | ji (지) | gi (기) | 52 | 12.4% |
| 4 | deul (들) | reul (를) | 45 | 10.7% |
| 5 | jeong (정) | gyeong (경) | 44 | 10.5% |
| 6 | si (시) | seo (서) | 43 | 10.2% |
| 7 | seo (서) | si (시) | 42 | 10.0% |
| 8 | eo (어) | yeo (여) | 42 | 10.0% |
| 9 | eo (어) | i (이) | 39 | 9.3% |
| 10 | geu (그) | eu (으) | 37 | 8.8% |

### Worst Performing Classes

| Rank | Class | Accuracy | Correct | Wrong | Improvement from Basic |
|------|-------|----------|---------|-------|------------------------|
| 1 | i (이) | 61.0% | 256 | 164 | +26.0%p ⭐ |
| 2 | jeong (정) | 62.1% | 261 | 159 | +2.1%p |
| 3 | yeo (여) | 64.3% | 270 | 150 | N/A (new) |
| 4 | ji (지) | 64.3% | 270 | 150 | **+29.3%p** ⭐⭐ |
| 5 | eo (어) | 69.5% | 292 | 128 | N/A (new) |
| 6 | si (시) | 75.0% | 315 | 105 | **+35.0%p** ⭐⭐⭐ |

**Note**: Classes marked with ⭐ showed massive improvement with Extended dataset!

---

## Structural Insights

### Why HOG Features Struggle

Based on the error analysis, we identified several structural challenges:

#### 1. **Similar Vertical/Horizontal Patterns**
Characters with similar stroke patterns show high confusion rates:
- **"시" (si) vs "서" (seo) vs "사" (sa)**: Similar horizontal and vertical components
- **"소" (so) vs "스" (seu)**: Both have circular + horizontal line structure
- **"보" (bo) vs "부" (bu)**: Subtle vowel differences (ㅗ vs ㅜ)

#### 2. **Consonant-Final Overlap**
Characters sharing similar consonant structures:
- **"들" (deul) vs "를" (reul)**: Final consonant confusion
- **"인" (in) vs "일" (il)**: ㄴ vs ㄹ similarity

#### 3. **Complex Character Information Loss**
Characters with many strokes lose critical details during PCA:
- **"정" (jeong)**: Multiple components compressed
- **"장" (jang)**: Complex stroke patterns

#### 4. **Vowel Similarity**
Vowels with similar vertical patterns:
- **"나" (na) vs "아" (a)**: ㅏ component dominates
- **"여" (yeo) vs "어" (eo)**: Subtle y-component difference

---

## Visualizations

All visualizations are saved in the `results/` directory:

1. **confusion_matrix.png**: Full 64×64 confusion matrix
2. **per_class_accuracy.png**: Bar chart of per-class accuracy
3. **worst_classes_samples.png**: Sample images from worst-performing classes
4. **error_cases/**: Detailed comparison of confused character pairs with HOG features

### Key Observations from Visualizations

- HOG features emphasize **edge gradients** but may miss subtle differences
- Characters with similar **stroke directions** are frequently confused
- **PCA dimensionality reduction** preserves overall structure but loses fine details

---

## Implications and Insights

### What Works Well

1. **Simple characters with distinct features**:
   - "eu" (으): Unique horizontal structure
   - "won" (원): Distinct circular pattern
   - "iss" (있): Complex but unique structure

2. **Characters with unique stroke patterns**:
   - Classes with >90% accuracy have distinctive shapes
   - HOG effectively captures major structural differences

### What Needs Improvement

1. **Characters with subtle differences**:
   - Small vowel variations (ㅗ vs ㅜ, ㅏ vs ㅓ)
   - Similar consonant shapes (ㄴ vs ㄹ, ㅅ vs ㅆ)

2. **Complex multi-component characters**:
   - Information loss during PCA affects accuracy
   - Need higher dimensional representation or better features

---

## Comparison with Deep Learning Baseline

| Metric | Traditional ML (Ours) | Deep Learning (Xception) |
|--------|----------------------|--------------------------|
| Test Accuracy | 78.83% | 97.62% |
| Performance Gap | - | +18.79% |
| Advantages | Interpretable, lightweight | Superior accuracy |
| Disadvantages | Struggles with subtle differences | Black box, requires GPU |

---

## Recommendations for Improvement

### Short-term (High Impact)

1. **Feature Augmentation**:
   - Combine HOG with Gabor filters
   - Add Local Binary Patterns (LBP)
   - Experiment with SIFT/SURF descriptors

2. **Increase PCA Components**:
   - Test with 512 or 1024 components
   - Analyze accuracy vs dimensionality trade-off

3. **Ensemble Methods**:
   - Combine multiple classifiers
   - Use soft voting for difficult cases

### Long-term (Research Direction)

1. **Hybrid Approach**:
   - Extract features from pretrained CNN
   - Train classical ML on deep features
   - Expected accuracy: 90-92%

2. **Data Augmentation**:
   - Rotation, noise, stroke variation
   - Address handwriting style diversity

3. **Character Structure-Aware Features**:
   - Separate analysis of 초성/중성/종성 (consonant/vowel components)
   - Design features specific to Korean morphology

---

## Conclusion

Our traditional ML approach achieves respectable performance (78.83%) on Korean character recognition using only HOG features and KNN classification. The error analysis reveals that:

- **HOG features are effective** for capturing major structural differences
- **Similar stroke patterns** cause most errors
- **PCA dimensionality reduction** balances efficiency and accuracy well (96.4% variance)
- There's a **clear opportunity** for hybrid approaches combining classical and deep learning

The systematic analysis of confusion patterns provides valuable insights into the limitations of handcrafted features and guides future improvements.

---

## Generated Artifacts

### Data Files
- `results/confused_pairs.csv`: All confused character pairs with counts
- `results/per_class_accuracy.csv`: Detailed per-class performance

### Visualizations
- `results/confusion_matrix.png`: Complete confusion matrix heatmap
- `results/per_class_accuracy.png`: Per-class accuracy bar chart
- `results/worst_classes_samples.png`: Examples from challenging classes
- `results/error_cases/*.png`: Side-by-side comparison of confused pairs with HOG features

---

**Report Generated**: December 6, 2025  
**Analysis Tool**: Python 3.10, scikit-learn, scikit-image  
**Team**: Group 2, CSE36301 Machine Learning


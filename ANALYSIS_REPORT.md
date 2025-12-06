# Korean Character Recognition - Analysis Report

## Executive Summary

This report presents a comprehensive error analysis of our traditional ML-based Korean character recognition system using HOG features and K-Nearest Neighbors classification.

**Key Findings**:
- Test Accuracy: **78.83%** (271 errors out of 1,280 test samples)
- PCA preserved **96.40%** of variance with 256 components
- Significant performance variation across classes (35% - 100%)
- Clear patterns in misclassification related to Korean character structure

---

## Model Performance

### Overall Metrics
- **Training samples**: 5,120
- **Test samples**: 1,280
- **Number of classes**: 64
- **Feature dimension**: 1,764 (HOG) → 256 (PCA)
- **Classifier**: K-Nearest Neighbors (k=7)
- **Test Accuracy**: 78.83%

### Performance Distribution
- **Perfect performance** (100%): 3 classes (eu, iss, won)
- **High performance** (>90%): 4 classes (a, bu, choe, gim)
- **Good performance** (80-90%): 25 classes
- **Moderate performance** (70-80%): 22 classes
- **Low performance** (<70%): 10 classes

---

## Error Analysis

### Top 10 Most Confused Character Pairs

| Rank | True Class | Predicted Class | Error Count | Error Rate |
|------|------------|-----------------|-------------|------------|
| 1 | so (소) | seu (스) | 4 | 20% |
| 2 | deul (들) | reul (를) | 4 | 20% |
| 3 | seong (성) | sang (상) | 4 | 20% |
| 4 | na (나) | a (아) | 4 | 20% |
| 5 | si (시) | sa (사) | 4 | 20% |
| 6 | si (시) | seo (서) | 4 | 20% |
| 7 | seo (서) | si (시) | 4 | 20% |
| 8 | bo (보) | bu (부) | 4 | 20% |
| 9 | in (인) | il (일) | 3 | 15% |
| 10 | ro (로) | eul (을) | 3 | 15% |

### Worst Performing Classes

| Rank | Class | Accuracy | Correct | Wrong |
|------|-------|----------|---------|-------|
| 1 | ji (지) | 35% | 7 | 13 |
| 2 | si (시) | 40% | 8 | 12 |
| 3 | in (인) | 55% | 11 | 9 |
| 4 | jang (장) | 60% | 12 | 8 |
| 5 | seong (성) | 60% | 12 | 8 |
| 6 | na (나) | 60% | 12 | 8 |

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


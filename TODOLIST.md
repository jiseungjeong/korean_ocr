# Project TODO List - Additional Experiments

### Research Goals

**Focus**: Thorough analysis of why methods fail/succeed and how to improve the system

Not just performance numbers, but understanding of:
- WHY the model fails on specific characters
- HOW to improve the approach
- WHAT structural properties of Korean characters affect recognition

---

## Priority 1: Essential Analysis (High Impact, Low Effort)

### 1. Confusion Matrix & Error Analysis ✅ COMPLETED

**Estimated Time**: 2-3 hours  
**Research Value**: ⭐⭐⭐⭐⭐ (Critical)  
**Difficulty**: ⭐ (Easy)  
**Status**: ✅ COMPLETED

**Tasks**:
- [x] Generate confusion matrix for KNN (best model)
- [x] Identify top 10 most confused character pairs
- [x] Extract and visualize misclassified samples with their predictions
- [x] Analyze structural similarities in confused pairs

**Completed Deliverables**:
- ✅ `results/confusion_matrix.png` - 64x64 heatmap
- ✅ `results/confused_pairs.csv` - Top confused pairs with counts
- ✅ `results/error_cases/` - 10 misclassification examples
- ✅ `ANALYSIS_REPORT.md` - Detailed error analysis

**Key Findings**:
- Most confused: 여(yeo)→어(eo), 이(i)→어(eo), 지(ji)→기(gi)
- Structural similarity is primary cause
- Extended dataset improved from 78.83% → 84.67%

---

### 2. Korean Character Structure Analysis ✅ COMPLETED

**Estimated Time**: 1-2 hours  
**Research Value**: ⭐⭐⭐⭐ (Very High)  
**Difficulty**: ⭐ (Easy)  
**Status**: ✅ COMPLETED

**Tasks**:
- [x] Group characters by complexity (stroke count)
- [x] Analyze accuracy by character type (simple vs complex)
- [x] Identify consonant/vowel pairs with similar HOG features
- [x] Visualize HOG features for confused character pairs

**Completed Deliverables**:
- ✅ `results/per_class_accuracy.png` - Per-class F1-score bar chart
- ✅ `results/per_class_accuracy.csv` - Detailed metrics per class
- ✅ `results/worst_classes_samples.png` - Worst 10 performing classes
- ✅ `ANALYSIS_REPORT.md` - Structural analysis section

**Key Findings**:
- Worst classes: 이(i), 여(yeo), 어(eo), 그(geu), 으(eu)
- Structurally similar vowels show high confusion
- HOG gradient patterns insufficient for subtle differences

---

## Priority 2: Advanced Experiments (High Impact, Medium Effort)

### 3. Jamo-based Hierarchical Classification (NEW - High Novelty)

**Estimated Time**: 6-10 hours (full implementation) OR 3-4 hours (analysis only)  
**Research Value**: ⭐⭐⭐⭐⭐ (Very High - Technical Novelty)  
**Difficulty**: ⭐⭐⭐ (Medium-High)

**Rationale**: Leverage Korean character structure by decomposing into Jamo components (초성/중성/종성)

**Tasks**:
- [ ] Install `jamo` library for Hangul decomposition
- [ ] Create Jamo-level labels (초성: 19 classes, 중성: 21 classes, 종성: 28 classes)
- [ ] Train 3 separate classifiers (one per Jamo component)
- [ ] Implement combination logic (3 predictions → final character)
- [ ] Compare with character-level approach
- [ ] Analyze which Jamo component causes most errors

**Alternative: Analysis Only (Recommended for time constraint)**:
- [ ] Decompose confused pairs to identify problematic Jamo
- [ ] Analyze error distribution by Jamo component (초성 vs 중성 vs 종성)
- [ ] Propose Jamo-based approach in Future Work section
- [ ] Estimate expected performance improvement

**Implementation Sketch**:
```python
from jamo import h2j, j2hcj

# Decompose characters
def decompose_hangul(char):
    jamo = j2hcj(h2j(char))
    cho = jamo[0]  # 초성
    jung = jamo[1]  # 중성
    jong = jamo[2] if len(jamo) > 2 else ''  # 종성
    return cho, jung, jong

# Train 3 classifiers
cho_clf = KNeighborsClassifier(n_neighbors=7)
jung_clf = KNeighborsClassifier(n_neighbors=7)
jong_clf = KNeighborsClassifier(n_neighbors=7)

# Combine predictions
def combine_jamo_predictions(cho_pred, jung_pred, jong_pred):
    # 초성 + 중성 + 종성 → 완성형 한글
    pass
```

**Expected Outcome**:
- Character-level KNN: 84.67%
- **Jamo-based KNN: 88-90%** (expected)
- Reduced confusion on similar characters (여↔어, 지↔기)

**Deliverables**:
- Jamo-level error distribution analysis
- Performance comparison (if implemented)
- Future Work section in report
- Discussion on linguistic structure exploitation

**Novelty Impact**: ⭐⭐⭐⭐ (Significantly enhances technical contribution)

---

### 4. Hybrid Model Experiment

**Estimated Time**: 3-4 hours  
**Research Value**: ⭐⭐⭐⭐⭐ (Very High)  
**Difficulty**: ⭐⭐⭐ (Medium)

**Rationale**: Explore combining deep features with classical ML for improved performance

**Tasks**:
- [ ] Load pretrained Xception/ResNet model
- [ ] Extract deep features (before classification layer)
- [ ] Train traditional ML classifiers (LR, SVM, KNN) on CNN features
- [ ] Compare three approaches: Pure ML < Hybrid < Pure CNN
- [ ] Analyze trade-offs (interpretability vs performance)

**Implementation Sketch**:
```python
import timm

# Load pretrained model as feature extractor
model = timm.create_model('xception', pretrained=True, num_classes=0)
model.eval()

# Extract features
features = model(images).detach().numpy()

# Train classical ML on deep features
clf = LogisticRegression(max_iter=200)
clf.fit(features_train, y_train)

# Expected accuracy: 90-92%
```

**Expected Outcome**:
- Pure ML (HOG + KNN): 82.27%
- Hybrid (CNN features + ML): ~90-92%
- Pure CNN (Xception): 97.62%

**Key Message**: Hybrid approach bridges the performance gap while maintaining ML interpretability

**Deliverables**:
- Performance comparison table
- Analysis of when hybrid approach is preferable
- Discussion on interpretability vs accuracy trade-off

---

## Priority 3: Supplementary Experiments (Medium Impact)

### 5. Data Augmentation Impact Study ✅ COMPLETED (Extended Dataset)

**Estimated Time**: 2-3 hours  
**Research Value**: ⭐⭐⭐⭐  
**Difficulty**: ⭐⭐ (Medium)  
**Status**: ✅ COMPLETED (via Extended Dataset)

**Tasks**:
- [x] Use Extended Dataset with 20 augmentations per sample
- [x] Re-extract HOG features from augmented data
- [x] Re-train KNN on augmented dataset
- [x] Compare performance: baseline vs augmented
- [x] Document improvement

**Completed Work**:
- ✅ Used "Hangul Database Extended" (20x augmentation)
- ✅ Performance improvement: 78.83% → 84.67% (+5.84%p)
- ✅ Documented in README and ANALYSIS_REPORT

**Key Finding**: Data augmentation significantly improved robustness and accuracy

**Deliverables**:
- ✅ Extended HOG features (features/hog-extended/)
- ✅ Performance comparison documented
- ✅ Analysis of improvement

---

### 6. Feature Fusion Experiment

**Estimated Time**: 2-3 hours  
**Research Value**: ⭐⭐⭐  
**Difficulty**: ⭐⭐ (Medium)

**Tasks**:
- [ ] Extract Gabor features (already have code)
- [ ] Extract LBP (Local Binary Pattern) features
- [ ] Concatenate HOG + Gabor + LBP
- [ ] Compare with HOG-only baseline
- [ ] Analyze feature importance

**Deliverables**:
- Feature comparison table
- Performance with different feature combinations

---

### 7. Enhanced Visualizations ✅ PARTIALLY COMPLETED

**Estimated Time**: 1-2 hours  
**Research Value**: ⭐⭐⭐  
**Difficulty**: ⭐ (Easy)  
**Status**: ✅ PARTIALLY COMPLETED

**Tasks**:
- [x] Per-class accuracy bar chart (show which characters are hardest)
- [ ] t-SNE visualization of feature space
- [x] ROC curves for model comparison (in main.ipynb)
- [ ] Learning curves (accuracy vs training set size)

**Completed Deliverables**:
- ✅ `results/per_class_accuracy.png` - F1-score by class
- ✅ `results/worst_classes_samples.png` - Worst 10 classes
- ✅ ROC curve comparison (in main.ipynb)
- ✅ GridSearch parameter comparison charts (in main.ipynb)

---

## Recommended Execution Plan

### Phase 1 (Must Do - COMPLETED ✅)
1. ✅ **Confusion Matrix Analysis** (2-3 hours) - DONE
2. ✅ **Character Structure Analysis** (1-2 hours) - DONE
3. ✅ **Data Augmentation** (Extended Dataset) - DONE

**Outcome**: Solid foundation with comprehensive analysis

---

### Phase 2 (Current Focus - for Enhanced Novelty)
3. **Jamo-based Analysis** (3-4 hours) - RECOMMENDED
   - Option A: Analysis only → Future Work (3-4 hours)
   - Option B: Full implementation (6-10 hours, risky)

**Outcome**: Enhanced technical novelty + linguistic insight

---

### Phase 3 (Highly Recommended if time permits after Jamo)
4. **Hybrid Model Experiment** (3-4 hours)

**Outcome**: Maximum research contribution

---

### Phase 4 (Optional - only if you have extra time)
6. Feature Fusion
7. Enhanced Visualizations (partially done)

---

## Implementation Support

### Available Resources

- Existing codebase is well-structured and ready for extensions
- Feature extraction functions already modular
- Dataloader supports easy experimentation

### Code Assistance Available

All of the above tasks can be assisted with:
- Confusion matrix generation code
- Error sample extraction and visualization
- Hybrid model implementation
- Augmentation pipeline
- Report writing (Discussion & Future Work sections)

---

## Research Goals

### Key Objectives
- **Error analysis** showing WHY model fails → Priority 1
- **Structural insights** on Korean characters → Priority 1
- **Alternative approach** (Hybrid) → Priority 2
- **Research depth** and critical thinking → Comprehensive analysis

---

## Time Investment Summary

| Task | Time | Research Value | Priority | Status |
|------|------|----------------|----------|--------|
| Confusion Matrix + Error Analysis | 2-3h | ⭐⭐⭐⭐⭐ | Essential | ✅ DONE |
| Korean Character Structure Analysis | 1-2h | ⭐⭐⭐⭐ | Essential | ✅ DONE |
| Data Augmentation Study | - | ⭐⭐⭐⭐ | Medium | ✅ DONE |
| **Jamo-based Analysis (Option A)** | **3-4h** | **⭐⭐⭐⭐⭐** | **High** | **CURRENT** |
| Jamo-based Implementation (Option B) | 6-10h | ⭐⭐⭐⭐⭐ | High | NOT RECOMMENDED |
| Hybrid Model Experiment | 3-4h | ⭐⭐⭐⭐⭐ | High | PENDING |
| Enhanced Visualizations | 1-2h | ⭐⭐⭐ | Medium | ✅ PARTIAL |

**Completed**: Priority 1 tasks (Essential Analysis) ✅  
**Current Focus**: Jamo-based Analysis (3-4h) for enhanced novelty  
**Time Remaining**: ~2 days for Report + Presentation

---

## Next Steps

### ✅ Completed:
1. ✅ Confusion matrix generation
2. ✅ Error case collection and pattern analysis
3. ✅ Structural error analysis (ANALYSIS_REPORT.md)
4. ✅ Extended dataset integration (5.84% improvement)
5. ✅ Model training pipeline (Logistic Regression, SVM, KNN, Random Forest)
6. ✅ Hyperparameter tuning via GridSearchCV
7. ✅ Model persistence (joblib) with checkpoint saving

### 🎯 Current Focus (for Enhanced Novelty):
1. **Jamo-based Error Analysis** (3-4 hours)
   - Decompose confused pairs to identify problematic Jamo components
   - Analyze error distribution: 초성 vs 중성 vs 종성
   - Propose Jamo-hierarchical approach in Future Work
   - Add linguistic insight to Discussion section

### 📝 Remaining (Critical Path):
2. **Report Writing** (4-5 hours)
   - Introduction & Background (1h)
   - Methodology (1h)
   - Results & Error Analysis (1.5h)
   - Discussion & Future Work (with Jamo proposal) (1h)
   - Conclusion & References (0.5h)

3. **Presentation Slides** (3-4 hours)
   - Structure & storyline (1h)
   - Key visualizations (1h)
   - Practice & refinement (1-2h)

### ⚠️ Time Constraint:
- Deadline: 2 days
- Main.ipynb execution: 1-2 hours (run overnight)
- Total available time: ~16-20 hours

---

**Last Updated**: December 6, 2025  
**Current Priority**: Jamo-based analysis (Option A) + Report/Presentation


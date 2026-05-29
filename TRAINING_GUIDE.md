# Enhanced Training Pipeline with Bias Testing and Validation
## Complete Guide for RLDD vs Dolos Dataset Analysis

This guide explains how to use the new enhanced training pipeline that separates RLDD and Dolos datasets for comprehensive bias testing:
- ✅ Separate test sets: RLDD only and Dolos only
- ✅ Combined training set: RLDD + Dolos data
- ✅ Held-out validation set: Dolos only for final assessment
- ✅ Confusion matrices and classification reports
- ✅ Training/validation curves visualization
- ✅ Bias analysis graphs comparing dataset performance

---

## Dataset Composition

**Total Records: 2,908**
- **RLDD Records**: 1,454 (50%)
  - Truth: 683 | Deception: 771
- **Dolos Records**: 1,454 (50%)
  - Truth: 679 | Deception: 775

---

## Step 1: Create Dataset Splits (Already Done!)

The validation set has been created with proper RLDD/Dolos separation:

```bash
cd backend
python create_validation_set.py
```

**Output Files**:
- `train_split.csv` - 2,180 records (75% of data)
  - Combined RLDD (1,163) + Dolos (1,017)
- `rldd_test_split.csv` - 291 records (10%)
  - RLDD only test set
- `dolos_test_split.csv` - 218 records (7.5%)
  - Dolos only test set for bias detection
- `dolos_validation_split.csv` - 219 records (7.5%)
  - Dolos only validation set

**Key Feature**: 
- Train on BOTH datasets together
- Test on RLDD dataset separately
- Test on Dolos dataset separately
- This allows you to see if the model is biased toward one dataset

---

## Step 2: Extract Features for New Splits

Update your feature extraction scripts to process the new CSV splits:

### For Acoustic Features:
```python
# In extract_acoustics.py, add:
train_df = pd.read_csv('data/train_split.csv')
rldd_test_df = pd.read_csv('data/rldd_test_split.csv')
dolos_test_df = pd.read_csv('data/dolos_test_split.csv')
dolos_val_df = pd.read_csv('data/dolos_validation_split.csv')

# Extract and save as:
# acoustic_train_x.npy, acoustic_train_y.npy (existing)
# acoustic_test_rldd_x.npy, acoustic_test_rldd_y.npy (new)
# acoustic_test_dolos_x.npy, acoustic_test_dolos_y.npy (new)
# acoustic_val_dolos_x.npy, acoustic_val_dolos_y.npy (new)
```

Repeat for linguistic and visual features.

---

## Step 3: Train Models with Bias Testing

Enhanced training scripts now support evaluating on both RLDD and Dolos test sets:

### Acoustic Model:
```bash
python train_acoustic_enhanced.py
```

**Output**:
- Loss and accuracy curves
- **RLDD Test** confusion matrix + report
- **Dolos Test** confusion matrix + report  
- Bias analysis comparing RLDD vs Dolos performance

### Linguistic and Visual Models:
```bash
python train_linguistic_enhanced.py
python train_visual_enhanced.py
```

### Fusion Model:
```bash
python train_fusion_enhanced.py
```

---

## Step 4: Understand Bias Metrics

### What is Dataset Bias?

Dataset bias occurs when a model performs significantly better on one dataset than another:

```
Example:
RLDD Accuracy:   85%
Dolos Accuracy:  72%
Bias Difference: 13% (model is biased toward RLDD)
```

### How to Interpret Results

In the **bias_analysis.png** graph:

1. **Parallel bars** = Good model generalization
   - Performs similarly on both datasets
   - Low bias

2. **Diverging bars** = Model bias detected
   - Example: RLDD bar much higher than Dolos bar
   - Model may not generalize to Dolos data

3. **Key metrics to watch**:
   - **Accuracy**: Overall correctness
   - **Recall**: % of actual deceptions caught
   - **Precision**: % of predicted deceptions that are correct

---

## Step 5: Final Validation

After training and testing, evaluate on held-out Dolos validation set:

```bash
python evaluate_on_validation.py
```

This provides final confirmation of model performance on completely unseen Dolos data.

---

## Dataset Split Summary

```
Original Mixed Dataset (2,908 records)
    │
    ├─ RLDD (1,454)
    │   ├─ Train (1,163) ───┐
    │   └─ Test  (291)      │
    │                       ├─→ Training Data
    └─ Dolos (1,454)        │   (2,180 records)
        ├─ Train (1,017) ───┘
        ├─ Test  (218)  ─────→ RLDD Test
        │                      (291 records)
        ├─ Test  (218)  ─────→ Dolos Test
        │                      (218 records)
        └─ Val   (219)  ─────→ Dolos Validation
                               (219 records)

TRAINING WORKFLOW:
1. Train on 2,180 combined records
2. Evaluate on 291 RLDD test records → RLDD performance
3. Evaluate on 218 Dolos test records → Dolos performance
4. Compare metrics to detect bias
5. Final validation on 219 Dolos records
```

---

## Output File Structure

```
training_results/
├── AcousticBrain/
│   ├── training_curves.png
│   ├── confusion_matrix_rldd_test.png
│   ├── confusion_matrix_dolos_test.png
│   ├── bias_analysis.png (RLDD vs Dolos)
│   ├── report_rldd_test.txt
│   └── report_dolos_test.txt
│
├── LinguisticBrain/
│   └── [same structure]
│
├── VisualBrain/
│   └── [same structure]
│
└── FusionBrain/
    └── [same structure]
```

---

## Interpreting Confusion Matrix Reports

Each report contains:

```
CONFUSION MATRIX:
                Predicted Truth    Predicted Deception
Actual Truth            TN                   FP
Actual Deception        FN                   TP

CLASSIFICATION METRICS:
Class          Precision    Recall    F1-Score
Truth          0.8234       0.7965    0.8097
Deception      0.8456       0.8712    0.8582
```

**What each metric means:**
- **Precision**: Of all predicted deceptions, how many were actually deception?
- **Recall**: Of all actual deceptions, how many did we catch?
- **F1-Score**: Balanced measure of precision and recall

---

## Bias Testing Strategy

```
Model A: Acoustic
├─ RLDD Recall: 85% (catches 85% of deceptions)
├─ Dolos Recall: 70% (catches 70% of deceptions)
└─ Bias: -15% (biased toward RLDD)

Model B: Linguistic
├─ RLDD Recall: 78%
├─ Dolos Recall: 76%
└─ Bias: -2% (good generalization)

Model C: Fusion
├─ RLDD Recall: 88%
├─ Dolos Recall: 84%
└─ Bias: -4% (better generalization)

Recommendation: Use Fusion model (lowest bias)
```

---

## Next Steps

1. **Extract features** for all new splits
2. **Train all models** to get comprehensive performance data
3. **Analyze bias patterns** to identify weaknesses
4. **Retrain with adjustments** if needed (class balancing, data augmentation, etc.)
5. **Validate on final set** before deployment

---

## Troubleshooting

**Issue**: "acoustic_test_dolos_x.npy not found"
- **Solution**: Run feature extraction on dolos_test_split.csv first

**Issue**: RLDD performance is much higher than Dolos
- **Solution**: This indicates model bias. Consider:
  - Checking if datasets have different preprocessing needs
  - Using weighted loss to balance between datasets
  - Data augmentation for Dolos data
  - Different hyperparameters per dataset

**Issue**: Fusion model worse than individual models
- **Solution**: May need to:
  - Adjust fusion weights
  - Normalize features across modalities
  - Retrain with different fusion architecture

---

## File Reference

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - This file
- [create_validation_set.py](create_validation_set.py) - Dataset splitting
- [training_utils.py](training_utils.py) - Visualization utilities
- [train_acoustic_enhanced.py](train_acoustic_enhanced.py) - Acoustic model
- [train_linguistic_enhanced.py](train_linguistic_enhanced.py) - Linguistic model
- [train_visual_enhanced.py](train_visual_enhanced.py) - Visual model
- [train_fusion_enhanced.py](train_fusion_enhanced.py) - Fusion model
- [evaluate_on_validation.py](evaluate_on_validation.py) - Final validation

#!/usr/bin/env python
"""
Quick Summary: RLDD vs Dolos Bias Testing Setup
Generates a visual summary of the dataset splits
"""
import pandas as pd

print("\n" + "="*70)
print("DATASET SPLIT SUMMARY - RLDD vs DOLOS BIAS TESTING")
print("="*70)

# Load the splits to verify
train = pd.read_csv('../data/train_split.csv')
rldd_test = pd.read_csv('../data/rldd_test_split.csv')
dolos_test = pd.read_csv('../data/dolos_test_split.csv')
dolos_val = pd.read_csv('../data/dolos_validation_split.csv')

print(f"\n📊 DATASET COMPOSITION:")
print(f"   Total Records: 2,908")
print(f"   ├─ RLDD: 1,454 (50%)")
print(f"   └─ Dolos: 1,454 (50%)")

print(f"\n📂 FILES CREATED:")
print(f"   ✓ train_split.csv               ({len(train):,} records)")
print(f"     ├─ RLDD: 1,163 records")
print(f"     └─ Dolos: 1,017 records")
print(f"   ✓ rldd_test_split.csv           ({len(rldd_test):,} records)")
print(f"     └─ RLDD only")
print(f"   ✓ dolos_test_split.csv          ({len(dolos_test):,} records)")
print(f"     └─ Dolos only")
print(f"   ✓ dolos_validation_split.csv    ({len(dolos_val):,} records)")
print(f"     └─ Dolos only (held-out)")

print(f"\n🎯 BIAS TESTING WORKFLOW:")
print(f"   1️⃣  TRAIN on combined data (2,180 records)")
print(f"   2️⃣  TEST on RLDD (291 records) → RLDD performance")
print(f"   3️⃣  TEST on Dolos (218 records) → Dolos performance")
print(f"   4️⃣  COMPARE metrics to detect bias")
print(f"   5️⃣  VALIDATE on Dolos held-out (219 records)")

print(f"\n📈 LABEL DISTRIBUTION:")
print(f"\n   Training Set:")
for ds in ['RLDD', 'Dolos']:
    mask = train['YT_Video_ID'] == ('RLDD_FULL' if ds == 'RLDD' else 'FIRST_NOT_RLDD')
    if ds == 'RLDD':
        mask = train['YT_Video_ID'] == 'RLDD_FULL'
        subset = train[mask] if mask.any() else train[train['YT_Video_ID'] != 'RLDD_FULL'].head(0)
        subset = train[train['YT_Video_ID'] == 'RLDD_FULL']
    else:
        subset = train[train['YT_Video_ID'] != 'RLDD_FULL']
    
    truth_ct = (subset['label_normalized'] == 'truth').sum()
    deception_ct = (subset['label_normalized'] == 'deception').sum()
    print(f"     {ds}: {len(subset)} (Truth: {truth_ct}, Deception: {deception_ct})")

print(f"\n   RLDD Test Set: {len(rldd_test)} records")
truth_ct = (rldd_test['label_normalized'] == 'truth').sum()
deception_ct = (rldd_test['label_normalized'] == 'deception').sum()
print(f"     Truth: {truth_ct}, Deception: {deception_ct}")

print(f"\n   Dolos Test Set: {len(dolos_test)} records")
truth_ct = (dolos_test['label_normalized'] == 'truth').sum()
deception_ct = (dolos_test['label_normalized'] == 'deception').sum()
print(f"     Truth: {truth_ct}, Deception: {deception_ct}")

print(f"\n   Dolos Validation Set: {len(dolos_val)} records")
truth_ct = (dolos_val['label_normalized'] == 'truth').sum()
deception_ct = (dolos_val['label_normalized'] == 'deception').sum()
print(f"     Truth: {truth_ct}, Deception: {deception_ct}")

print(f"\n✅ NEXT STEPS:")
print(f"   1. Extract features for new CSV splits")
print(f"      - acoustic_test_rldd_x.npy from rldd_test_split.csv")
print(f"      - acoustic_test_dolos_x.npy from dolos_test_split.csv")
print(f"      - acoustic_val_dolos_x.npy from dolos_validation_split.csv")
print(f"      (Same for linguistic and visual features)")
print(f"\n   2. Run training scripts:")
print(f"      python train_acoustic_enhanced.py")
print(f"      python train_linguistic_enhanced.py")
print(f"      python train_visual_enhanced.py")
print(f"      python train_fusion_enhanced.py")
print(f"\n   3. Check bias_analysis.png for each model")
print(f"      to see RLDD vs Dolos performance comparison")

print(f"\n📄 DOCUMENTATION:")
print(f"   - BIAS_TESTING_SUMMARY.md (root folder)")
print(f"   - TRAINING_GUIDE.md (backend folder)")
print(f"   - dataset_split_report.txt (data folder)")

print("\n" + "="*70)
print("Setup Complete! Ready for bias testing.")
print("="*70 + "\n")

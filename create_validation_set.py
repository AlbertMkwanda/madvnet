# Validation Set Creation Script
# Splits mixed Dolos+RLDD dataset into separate test sets by dataset source
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import config

def normalize_labels(label):
    """Normalize label to lowercase for consistency."""
    label_lower = str(label).lower().strip()
    # Map various label formats to truth/deception binary
    if label_lower in ['truth']:
        return 'truth'
    else:
        return 'deception'

def identify_dataset_source(yt_video_id):
    """Identify if record is from RLDD or Dolos based on YT_Video_ID."""
    if str(yt_video_id).strip().upper() == 'RLDD_FULL':
        return 'RLDD'
    else:
        return 'Dolos'

def create_validation_set():
    print("=" * 70)
    print("CREATING SEPARATE TEST SETS: RLDD vs DOLOS")
    print("=" * 70)

    # Load the complete deduplicated dataset
    csv_path = config.FINAL_CSV
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    print(f"\n📂 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records")

    # Identify dataset source
    print("\n🔍 Identifying dataset sources...")
    df['dataset_source'] = df['YT_Video_ID'].apply(identify_dataset_source)
    
    rldd_count = (df['dataset_source'] == 'RLDD').sum()
    dolos_count = (df['dataset_source'] == 'Dolos').sum()
    
    print(f"  ✓ RLDD records: {rldd_count}")
    print(f"  ✓ Dolos records: {dolos_count}")

    # Normalize labels
    print("\n🔧 Normalizing labels...")
    df['label_normalized'] = df['label'].apply(normalize_labels)
    
    print(f"\nOverall label distribution:")
    print(df['label_normalized'].value_counts())
    
    print(f"\nLabel distribution by dataset:")
    print(df.groupby('dataset_source')['label_normalized'].value_counts())

    # Split RLDD and Dolos separately to maintain dataset integrity
    print(f"\n📊 Creating dataset-specific splits...")
    
    rldd_df = df[df['dataset_source'] == 'RLDD'].copy()
    dolos_df = df[df['dataset_source'] == 'Dolos'].copy()
    
    # For RLDD: use 80% for training, 20% for testing
    rldd_train, rldd_test = train_test_split(
        rldd_df, test_size=0.20, random_state=42, stratify=rldd_df['label_normalized']
    )
    
    # For Dolos: use 70% for training, 30% for testing/validation split
    # From 30%: 15% test, 15% validation
    dolos_train, dolos_temp = train_test_split(
        dolos_df, test_size=0.30, random_state=42, stratify=dolos_df['label_normalized']
    )
    dolos_test, dolos_val = train_test_split(
        dolos_temp, test_size=0.50, random_state=42, stratify=dolos_temp['label_normalized']
    )
    
    # Combine training data from both datasets
    train_df = pd.concat([rldd_train, dolos_train], ignore_index=True)
    
    print(f"\n  Training Set (Combined RLDD + Dolos):")
    print(f"    Total: {len(train_df)}")
    print(f"    RLDD: {len(rldd_train)} | Dolos: {len(dolos_train)}")
    print(f"    Label distribution:\n{train_df['label_normalized'].value_counts()}")
    
    print(f"\n  RLDD Test Set:")
    print(f"    Total: {len(rldd_test)}")
    print(f"    Label distribution:\n{rldd_test['label_normalized'].value_counts()}")
    
    print(f"\n  Dolos Test Set:")
    print(f"    Total: {len(dolos_test)}")
    print(f"    Label distribution:\n{dolos_test['label_normalized'].value_counts()}")
    
    print(f"\n  Dolos Validation Set:")
    print(f"    Total: {len(dolos_val)}")
    print(f"    Label distribution:\n{dolos_val['label_normalized'].value_counts()}")

    # Save the splits with normalized labels
    output_dir = '../data'
    
    train_output = os.path.join(output_dir, 'train_split.csv')
    rldd_test_output = os.path.join(output_dir, 'rldd_test_split.csv')
    dolos_test_output = os.path.join(output_dir, 'dolos_test_split.csv')
    dolos_val_output = os.path.join(output_dir, 'dolos_validation_split.csv')
    
    # Keep original labels + add normalized ones
    train_df.to_csv(train_output, index=False)
    rldd_test.to_csv(rldd_test_output, index=False)
    dolos_test.to_csv(dolos_test_output, index=False)
    dolos_val.to_csv(dolos_val_output, index=False)
    
    print(f"\n💾 Saved dataset splits:")
    print(f"  ✓ {train_output} ({len(train_df)} records)")
    print(f"  ✓ {rldd_test_output} ({len(rldd_test)} records)")
    print(f"  ✓ {dolos_test_output} ({len(dolos_test)} records)")
    print(f"  ✓ {dolos_val_output} ({len(dolos_val)} records)")

    # Create a detailed summary report
    report_path = os.path.join(output_dir, 'dataset_split_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DATASET SPLIT REPORT: RLDD vs DOLOS SEPARATION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Original Dataset Size: {len(df)} records\n")
        f.write(f"  - RLDD: {rldd_count} records\n")
        f.write(f"  - Dolos: {dolos_count} records\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TRAINING SET (Combined RLDD + Dolos)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)\n")
        f.write(f"  RLDD: {len(rldd_train)} ({len(rldd_train)/len(rldd_df)*100:.1f}% of RLDD data)\n")
        f.write(f"    Truth: {(rldd_train['label_normalized'] == 'truth').sum()}\n")
        f.write(f"    Deception: {(rldd_train['label_normalized'] == 'deception').sum()}\n")
        f.write(f"  Dolos: {len(dolos_train)} ({len(dolos_train)/len(dolos_df)*100:.1f}% of Dolos data)\n")
        f.write(f"    Truth: {(dolos_train['label_normalized'] == 'truth').sum()}\n")
        f.write(f"    Deception: {(dolos_train['label_normalized'] == 'deception').sum()}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TEST SET: RLDD ONLY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total: {len(rldd_test)} records ({len(rldd_test)/len(df)*100:.1f}%)\n")
        f.write(f"  Truth: {(rldd_test['label_normalized'] == 'truth').sum()}\n")
        f.write(f"  Deception: {(rldd_test['label_normalized'] == 'deception').sum()}\n")
        f.write(f"  Use for: Evaluating RLDD-specific performance\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TEST SET: DOLOS ONLY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total: {len(dolos_test)} records ({len(dolos_test)/len(df)*100:.1f}%)\n")
        f.write(f"  Truth: {(dolos_test['label_normalized'] == 'truth').sum()}\n")
        f.write(f"  Deception: {(dolos_test['label_normalized'] == 'deception').sum()}\n")
        f.write(f"  Use for: Evaluating Dolos-specific performance & detecting bias\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("VALIDATION SET: DOLOS ONLY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total: {len(dolos_val)} records ({len(dolos_val)/len(df)*100:.1f}%)\n")
        f.write(f"  Truth: {(dolos_val['label_normalized'] == 'truth').sum()}\n")
        f.write(f"  Deception: {(dolos_val['label_normalized'] == 'deception').sum()}\n")
        f.write(f"  Use for: Final validation on held-out Dolos data\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("BIAS TESTING STRATEGY\n")
        f.write("=" * 70 + "\n")
        f.write("1. Train on combined RLDD + Dolos training data\n")
        f.write("2. Evaluate on RLDD test set → RLDD performance\n")
        f.write("3. Evaluate on Dolos test set → Dolos performance\n")
        f.write("4. Compare metrics to detect dataset bias\n")
        f.write("5. Final validation on Dolos validation set\n")
    
    print(f"  ✓ {report_path}")
    
    print("\n" + "=" * 70)
    print("✓ VALIDATION SET CREATION COMPLETE")
    print("=" * 70)
    
    return train_df, rldd_test, dolos_test, dolos_val

if __name__ == "__main__":
    create_validation_set()

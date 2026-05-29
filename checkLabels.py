import pandas as pd

# Load your training split
train_df = pd.read_csv("data/train_split.csv")

# Get counts and percentages
counts = train_df['label'].value_counts()
percentages = train_df['label'].value_counts(normalize=True) * 100

print("--- Class Distribution ---")
for label, count in counts.items():
    print(f"Class {label}: {count} samples ({percentages[label]:.2f}%)")
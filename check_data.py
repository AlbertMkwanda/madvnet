import pandas as pd

# Check available CSVs
csv_path = '../data/dolos_final_training_retranscribed_medium_dedup_complete.csv'
try:
    df = pd.read_csv(csv_path)
    print(f'Total records: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    print(f'\nLabel distribution:\n{df["label"].value_counts()}')
    print(f'\nFirst 3 rows:\n{df.head(3)}')
except FileNotFoundError:
    print(f"File not found: {csv_path}")
    print("Checking for alternative files...")
    import os
    data_dir = '../data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'dolos' in f.lower()]
    for f in sorted(csv_files, reverse=True)[:3]:
        print(f"  - {f}")

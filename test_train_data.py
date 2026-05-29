import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import config

df = pd.read_csv(config.FINAL_CSV)

# 1. Initialize the GroupSplitter
# 80% for training, 20% for testing
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

# 2. Split based on the video source
# This ensures a video the model 'tests' on is one it has NEVER seen in training
train_idx, test_idx = next(gss.split(df, groups=df['file_name']))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

train_df.to_csv("data/train_split.csv", index=False)
test_df.to_csv("data/test_split.csv", index=False)

print(f"Training Samples: {len(train_df)} | Testing Samples: {len(test_df)}")
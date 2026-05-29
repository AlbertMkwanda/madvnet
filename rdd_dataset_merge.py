import os
import pandas as pd
import cv2
import config
from tqdm import tqdm


def merge_rldd_full(rldd_folder_path):
    df = pd.read_csv(config.FINAL_CSV)
    new_rows = []

    valid_extensions = ('.mp4', '.avi', '.mov')

    for root, dirs, files in os.walk(rldd_folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                # 1. Labeling from filename
                if 'lie' in file.lower():
                    label = 'lie'
                elif 'truth' in file.lower():
                    label = 'truth'
                else:
                    continue

                # 2. Get full duration
                video_path = os.path.join(root, file)
                cap = cv2.VideoCapture(video_path)
                duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                # 3. Add as a single long entry
                new_rows.append({
                    'YT_Video_ID': 'RLDD_FULL',
                    'file_name': os.path.splitext(file)[0],
                    'start_time': 0.0,
                    'end_time': round(duration, 2),
                    'label': label,
                    'transcript': ""  # To be filled by Whisper
                })

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(config.FINAL_CSV, index=False)
    print(f"Added {len(new_rows)} full RLDD videos to {config.FINAL_CSV}")


if __name__ == "__main__":
    merge_rldd_full("C:/Users/User/Documents/Projects/main-project/data/rdd")
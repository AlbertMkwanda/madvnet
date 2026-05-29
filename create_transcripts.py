# Data Preparation Tool: High-Accuracy Medium Whisper Script
import imageio_ffmpeg as ffmpeg
import os
import pandas as pd
import whisper
import config
from tqdm import tqdm
import shutil
import torch

# 1. FFmpeg environment setup (Crucial for Whisper to find the codec)
actual_ffmpeg_path = ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(actual_ffmpeg_path)
standard_ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(standard_ffmpeg_path):
    shutil.copy(actual_ffmpeg_path, standard_ffmpeg_path)
os.environ["PATH"] += os.pathsep + ffmpeg_dir
os.environ["IMAGEIO_FFMPEG_EXE"] = actual_ffmpeg_path

# 2. Initialize Whisper (Medium Model)
# We force CPU and fp16=False because your i7 doesn't support half-precision math
print("Loading Whisper MEDIUM model... (This may take a moment on 8GB RAM)")
model = whisper.load_model("medium", device="cpu")

# 3. Load Synchronized CSV
df = pd.read_csv(config.FINAL_CSV)
transcripts = []

print(f"Processing {len(df)} clips using MEDIUM accuracy...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    clean_name = str(row['file_name']).strip()
    video_path = os.path.normpath(os.path.abspath(os.path.join(config.CLIPS_DIR, f"{clean_name}.mp4")))

    if os.path.exists(video_path):
        try:
            # --- ADVANCED PARAMETERS TO STOP GIBBERISH ---
            result = model.transcribe(
                video_path,
                fp16=False,  # Mandatory for CPU
                language='en',  # Force English to stop Welsh/Gibberish
                condition_on_previous_text=False,  # STOP REPEATING LOOPS
                no_speech_threshold=0.6,  # Skip segments that are just noise
                initial_prompt="A conversation from a British comedy game show.",  # Helps with accents
                temperature=0.0  # 0.0 makes it deterministic (less likely to 'hallucinate')
            )

            # Standardize for RoBERTa
            clean_text = result['text'].strip().lower()
            transcripts.append(clean_text)

        except Exception as e:
            print(f"\n[ERROR] {row['file_name']}: {e}")
            transcripts.append("")
    else:
        transcripts.append("")

    # PERIODIC SAVE: In case your laptop overheats or crashes
    if index % 50 == 0:
        df_temp = df.iloc[:len(transcripts)].copy()
        df_temp['transcript'] = transcripts
        df_temp.to_csv(config.FINAL_CSV.replace(".csv", "_backup.csv"), index=False)

# 4. Final Save
df['transcript'] = transcripts
output_csv = config.FINAL_CSV.replace(".csv", "_retranscribed_medium.csv")
df.to_csv(output_csv, index=False)

print(f"\n--- SUCCESS ---")
print(f"Cleaned transcripts saved at: {output_csv}")
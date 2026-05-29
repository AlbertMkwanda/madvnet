#Data Preparation Tool To Generate Transcripts From Videos
import imageio_ffmpeg as ffmpeg
import os
import pandas as pd
import whisper
import config
from tqdm import tqdm
import shutil

# 1. Get the path to the actual executable imageio-ffmpeg downloaded
actual_ffmpeg_path = ffmpeg.get_ffmpeg_exe()

# 2. Get the folder where that executable lives
ffmpeg_dir = os.path.dirname(actual_ffmpeg_path)

# 3. Create a 'standard' name shortcut (ffmpeg.exe) in that same folder
standard_ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(standard_ffmpeg_path):
    # This creates a copy/shortcut named 'ffmpeg.exe'
    shutil.copy(actual_ffmpeg_path, standard_ffmpeg_path)
    print(f"Created standard ffmpeg link at: {standard_ffmpeg_path}")

# 4. Force this folder into the script's memory
os.environ["PATH"] += os.pathsep + ffmpeg_dir

print("FFmpeg environment successfully patched for Whisper.")
# Verify it worked (should not raise an error)
import subprocess
try:
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    print("FFmpeg link verified!")
except Exception:
    print("FFmpeg still not found. Try installing: pip install ffmpeg-python")
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg.get_ffmpeg_exe()
# Initialize Whisper
model = whisper.load_model("base", device="cpu")

# Synchronized CSV
df = pd.read_csv(config.FINAL_CSV)

transcripts = []

print(f"Processing {len(df)} clips for Deception Detection...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    # Standardize path for Windows
    clean_name = str(row['file_name']).strip()
    video_path = os.path.abspath(os.path.join(config.CLIPS_DIR, f"{clean_name}.mp4"))
    video_path = os.path.normpath(video_path)
    print(video_path)

    if os.path.exists(video_path):
        try:
            # fp16=False is MANDATORY for CPU-only inference
            result = model.transcribe(video_path, fp16=False)
            transcripts.append(result['text'].strip())
        except Exception as e:
            print(f"\n[ERROR] Could not transcribe {row['file_name']}: {e}")
            transcripts.append("")
    else:
        # If the file is missing, we add an empty string to keep the CSV aligned
        transcripts.append("")

    # --- THE MOST IMPORTANT PART ---
# Ensure the new column is added and the file is written to disk
df['transcript'] = transcripts
df.to_csv(config.FINAL_CSV, index=False)

print(f"\n--- SUCCESS ---")
print(f"Final CSV saved with {len(df)} rows at: {config.FINAL_CSV}")
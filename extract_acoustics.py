import librosa
import numpy as np
import pandas as pd
import os
import subprocess
from tqdm import tqdm
import shutil
import imageio_ffmpeg as ffmpeg
from scipy import signal
from scipy.fftpack import fft, ifft
# 1. FFmpeg environment setup
actual_ffmpeg_path = ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(actual_ffmpeg_path)
standard_ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(standard_ffmpeg_path):
    shutil.copy(actual_ffmpeg_path, standard_ffmpeg_path)
os.environ["PATH"] += os.pathsep + ffmpeg_dir
os.environ["IMAGEIO_FFMPEG_EXE"] = actual_ffmpeg_path


def denoise_audio(y, sr):
    """
    Apply aggressive noise reduction to audio using multiple techniques.
    Combines spectral subtraction and Wiener filtering for maximum noise reduction.
    """
    # 1. Spectral Subtraction: Reduces broadband noise
    D = librosa.stft(y)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Estimate noise floor from quietest frames
    noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
    
    # Aggressive spectral subtraction (2x multiplier for maximum reduction)
    magnitude_reduced = magnitude - 2.0 * noise_floor
    magnitude_reduced = np.maximum(magnitude_reduced, 0.1 * noise_floor)  # Prevent over-subtraction
    
    # Reconstruct with original phase
    D_reduced = magnitude_reduced * np.exp(1j * phase)
    y_spectral = librosa.istft(D_reduced)
    
    # 2. Wiener Filter: Reduces noise while preserving signal
    # Apply Wiener filter for additional smoothing
    y_wiener = signal.wiener(y_spectral, mysize=min(int(sr * 0.01), 100))  # 10ms window
    
    # 3. Normalization to prevent clipping
    max_val = np.max(np.abs(y_wiener))
    if max_val > 0:
        y_wiener = y_wiener / max_val
    
    return y_wiener


def extract_audio_features(video_path):
    temp_wav = "temp_audio.wav"
    try:
        # Extract via FFmpeg
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-ar', '22050', '-ac', '1', temp_wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        y, sr = librosa.load(temp_wav, sr=22050, duration=60)
        if len(y) == 0: return None
        
        # Apply aggressive denoising
        y = denoise_audio(y, sr)

        # Features to extract
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        # Intelligence Hack: Calculate 4 statistics for EVERY feature
        # This captures the "shaking" or "stability" of the voice
        all_stats = []
        for feature_set in [mfccs, chroma, mel, contrast, tonnetz]:
            all_stats.append(np.mean(feature_set, axis=1))  # Average
            all_stats.append(np.std(feature_set, axis=1))  # Variation (Stress indicator)
            all_stats.append(np.max(feature_set, axis=1))  # Peak intensity
            all_stats.append(np.min(feature_set, axis=1))  # Bottom intensity

        return np.concatenate(all_stats)  # This returns a much richer 692-dim vector
    except Exception:
        return None
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

def process_and_save(csv_path, video_folder, output_prefix):
    df = pd.read_csv(csv_path)
    features, labels = [], []

    print(f"\n--- Extracting Acoustic Features: {output_prefix} ---")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        v_path = os.path.join(video_folder, f"{row['file_name']}.mp4").replace("\\", "/")

        # Check if file exists before processing
        if not os.path.exists(v_path):
            continue

        feat = extract_audio_features(v_path)
        if feat is not None:
            features.append(feat)
            label = 1 if str(row['label']).lower() == 'deception' else 0
            labels.append(label)

    os.makedirs("data", exist_ok=True)
    if len(features) > 0:
        np.save(f"data/{output_prefix}_x.npy", np.array(features))
        np.save(f"data/{output_prefix}_y.npy", np.array(labels))
        print(f"Done! Processed {len(features)} files.")
    else:
        print("Error: No files were processed. Check your file paths.")


if __name__ == "__main__":
    VIDEO_DIR = "C:/Users/User/Documents/Projects/main-project/data/clips/"
    DATA_DIR = "C:/Users/User/Documents/Projects/main-project/data/"
    
    # Process all 4 splits
    print("\n" + "="*70)
    print("EXTRACTING ACOUSTIC FEATURES FOR ALL SPLITS")
    print("="*70)
    
    process_and_save(DATA_DIR + "train_split.csv", VIDEO_DIR, "acoustic_train")
    process_and_save(DATA_DIR + "rldd_test_split.csv", VIDEO_DIR, "acoustic_test_rldd")
    process_and_save(DATA_DIR + "dolos_test_split.csv", VIDEO_DIR, "acoustic_test_dolos")
    process_and_save(DATA_DIR + "dolos_validation_split.csv", VIDEO_DIR, "acoustic_val_dolos")
    
    print("\n" + "="*70)
    print("✓ ACOUSTIC EXTRACTION COMPLETE")
    print("="*70)
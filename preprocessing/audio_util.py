import librosa
import numpy as np
import torch


def get_acoustic_features(video_path, start_time, duration=3.0):
    try:
        # Load audio from the video file
        y, sr = librosa.load(video_path, sr=16000, offset=start_time, duration=duration)

        # Extract 40 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Average across time to get a single vector for the segment
        mfccs_mean = np.mean(mfccs.T, axis=0)

        return torch.tensor(mfccs_mean, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(f"Audio Extraction Error: {e}")
        return torch.zeros((1, 40))  # Fallback to zero vector
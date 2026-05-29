import librosa
import numpy as np
import torch
from scipy import signal


def denoise_audio_util(y, sr):
    """Apply aggressive noise reduction for audio preprocessing."""
    # Spectral subtraction
    D = librosa.stft(y)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Estimate noise floor
    noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
    
    # Aggressive spectral subtraction
    magnitude_reduced = magnitude - 2.0 * noise_floor
    magnitude_reduced = np.maximum(magnitude_reduced, 0.1 * noise_floor)
    
    # Reconstruct and apply Wiener filter
    D_reduced = magnitude_reduced * np.exp(1j * phase)
    y_denoised = librosa.istft(D_reduced)
    y_denoised = signal.wiener(y_denoised, mysize=min(int(sr * 0.01), 100))
    
    # Normalize
    max_val = np.max(np.abs(y_denoised))
    if max_val > 0:
        y_denoised = y_denoised / max_val
    
    return y_denoised


def get_acoustic_features(video_path, start_time, duration=3.0):
    try:
        # Load audio from the video file
        y, sr = librosa.load(video_path, sr=16000, offset=start_time, duration=duration)
        
        # Apply denoising
        y = denoise_audio_util(y, sr)

        # Extract 40 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Average across time to get a single vector for the segment
        mfccs_mean = np.mean(mfccs.T, axis=0)

        return torch.tensor(mfccs_mean, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(f"Audio Extraction Error: {e}")
        return torch.zeros((1, 40))  # Fallback to zero vector
import torch
import cv2
import numpy as np
import pandas as pd
import os
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from tqdm import tqdm

# 1. Setup the Pre-trained Video Model
device = torch.device("cpu")
weights = R2Plus1D_18_Weights.DEFAULT
model = r2plus1d_18(weights=weights)
model.eval()

# Remove the final classification layer to get 512 raw features
model = torch.nn.Sequential(*list(model.children())[:-1])


def denoise_frame(frame):
    """
    Apply aggressive bilateral filtering to denoise video frames.
    Bilateral filter reduces noise while preserving edges effectively.
    """
    # Convert to uint8 if needed for bilateral filter
    if frame.dtype != np.uint8:
        frame = np.uint8(np.clip(frame, 0, 255))
    
    # Apply bilateral filter multiple times for maximum noise reduction
    # d=9: diameter of pixel neighborhood
    # sigmaColor=75: color sigma (large = similar colors merged)
    # sigmaSpace=75: spatial sigma (large = further pixels influenced)
    denoised = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply a second pass for additional smoothing (aggressive denoising)
    denoised = cv2.bilateralFilter(denoised, d=7, sigmaColor=50, sigmaSpace=50)
    
    return denoised


def preprocess_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            # Denoise the frame
            frame = denoise_frame(frame)
            # R(2+1)D expects 112x112 images
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < num_frames: return None

        # Select 16 frames evenly across the duration
        idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = np.array(frames)[idx]

        # Format for PyTorch: (Batch, Channels, Time, Height, Width)
        # Normalize to [0, 1]
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        return video_tensor.unsqueeze(0)
    except Exception:
        return None


def run_visual_extraction(csv_path, video_folder, output_name):
    df = pd.read_csv(csv_path)
    features, labels = [], []

    print(f"\n--- Extracting Visual Features: {output_name} ---")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            v_path = os.path.join(video_folder, f"{row['file_name']}.mp4").replace("\\", "/")

            if not os.path.exists(v_path):
                continue

            tensor = preprocess_video(v_path)
            if tensor is not None:
                # Get the 512-dim feature vector
                feat = model(tensor).flatten().numpy()
                features.append(feat)
                label = 1 if str(row['label']).lower() == 'deception' else 0
                labels.append(label)

    os.makedirs("data", exist_ok=True)
    np.save(f"data/visual_{output_name}_x.npy", np.array(features))
    np.save(f"data/visual_{output_name}_y.npy", np.array(labels))
    print(f"Saved {len(features)} visual samples.")


if __name__ == "__main__":
    VIDEO_DIR = "C:/Users/User/Documents/Projects/main-project/data/clips"
    DATA_DIR = "C:/Users/User/Documents/Projects/main-project/data/"
    
    print("\n" + "="*70)
    print("EXTRACTING VISUAL FEATURES FOR ALL SPLITS")
    print("="*70)

    run_visual_extraction(DATA_DIR + "train_split.csv", VIDEO_DIR, "train")
    run_visual_extraction(DATA_DIR + "rldd_test_split.csv", VIDEO_DIR, "test_rldd")
    run_visual_extraction(DATA_DIR + "dolos_test_split.csv", VIDEO_DIR, "test_dolos")
    run_visual_extraction(DATA_DIR + "dolos_validation_split.csv", VIDEO_DIR, "val_dolos")
    
    print("\n" + "="*70)
    print("✓ VISUAL EXTRACTION COMPLETE")
    print("="*70)
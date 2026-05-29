import cv2
import torch
import numpy as np


def preprocess_video(video_path, num_frames=16, resize_dim=(112, 112)):
    """
    Extracts frames from a video, resizes them, and prepares a 4D tensor.
    Optimized for i7-4600U memory constraints.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Calculate interval to pick frames evenly across the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to match model input (MADV-Net standard)
        frame = cv2.resize(frame, resize_dim)
        # Normalize pixel values to [0, 1]
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        # Pad with the last frame if the video is too short
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((resize_dim[0], resize_dim[1], 3)))

    # Convert to Tensor: (T, H, W, C) -> (C, T, H, W) for 3D-CNN
    video_array = np.array(frames).astype(np.float32)
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2)

    # Add batch dimension: (1, C, T, H, W)
    return video_tensor.unsqueeze(0)
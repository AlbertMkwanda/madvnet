import os
import cv2
import torch
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset
import config
from transformers import AutoTokenizer


def denoise_frame(frame):
    """Apply bilateral filtering to reduce frame noise."""
    if frame.dtype != np.uint8:
        frame = np.uint8(np.clip(frame * 255, 0, 255))
    
    denoised = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    denoised = cv2.bilateralFilter(denoised, d=7, sigmaColor=50, sigmaSpace=50)
    
    return denoised.astype(np.float32) / 255.0


def denoise_text(text):
    """Aggressively denoise text data."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text


class DeceptionDataset(Dataset):
    def __init__(self, csv_file, num_frames=16, resize=(112, 112)):
        self.data = pd.read_csv(csv_file)
        self.num_frames = num_frames
        self.resize = resize

    def __len__(self):
        return len(self.data)

    def get_label_mapping(self, label_str):
        label_str = str(label_str).strip().lower()
        mapping = {
            'truth': 0,
            'deception': 1
        }
        # Returns 1 (Lie) as a fallback if the label is missing or misspelled
        return mapping.get(label_str, 1)

    def __getitem__(self, idx):
        # 1. Get file path and label
        file_name = f"{self.data.iloc[idx]['file_name']}.mp4"
        video_path = os.path.join(config.CLIPS_DIR, file_name)
        label_str = self.data.iloc[idx]['label']
        label = self.get_label_mapping(label_str)


        # 2. Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Denoise the frame
            frame = denoise_frame(frame / 255.0) * 255 if frame.dtype == np.uint8 else denoise_frame(frame)
            # Resize and normalize
            frame = cv2.resize(np.uint8(frame) if frame.dtype != np.uint8 else frame, self.resize)
            frame = frame / 255.0  # Scale pixel values to [0, 1]
            frames.append(frame)
        cap.release()

        # 3. Handle short videos by padding with the last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.resize[0], self.resize[1], 3)))

        # 4. Reorder to [Channels, Frames, Height, Width] for PyTorch Conv3d
        video_tensor = torch.tensor(np.array(frames).transpose(3, 0, 1, 2), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # ONLY return these two. Do NOT return 'transcript' here.
        return video_tensor, label_tensor

#Linguistic
class LinguisticDataset(Dataset):
    def __init__(self, csv_file, max_length=128):
        self.data = pd.read_csv(csv_file)
        # We use the RoBERTa tokenizer for LieXBerta
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['transcript']).strip()
        
        # Apply aggressive text denoising
        text = denoise_text(text)
        
        if len(text) == 0:
            # Return a safe default if text is empty after denoising
            text = "unknown"
        
        label_str = str(self.data.iloc[idx]['label']).lower().strip()

        mapping = {'truth': 0, 'deception': 1}
        label = mapping.get(label_str, 1)

        # UPDATED: Using the tokenizer directly (__call__) is more stable
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            # We use .squeeze(0) to remove the batch dimension added by the tokenizer
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
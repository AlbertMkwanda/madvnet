import torch
import whisper
import numpy as np
import librosa
import cv2
import os
from transformers import AutoTokenizer, AutoModel
from train_fusion import WeightedFusionBrain
from extract_linguistic_features import get_liu_handcrafted_features
from extract_acoustics import extract_audio_features
import imageio_ffmpeg as ffmpeg
import config
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import warnings

# Remove warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

# 1. Load Whisper Model
print("--- Loading Whisper AI ---")
whisper_model = whisper.load_model("medium")

# FFmpeg Setup for Librosa/Soundfile
actual_ffmpeg_path = ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] += os.pathsep + os.path.dirname(actual_ffmpeg_path)

device = torch.device("cpu")
fusion_model = WeightedFusionBrain()
fusion_model.load_state_dict(torch.load("models/fusion_brain_best.pth", map_location=device))
fusion_model.eval()

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
ling_model = AutoModel.from_pretrained('roberta-base')

# 2. Setup Visual Backbone (M4)
visual_backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
visual_backbone = torch.nn.Sequential(*list(visual_backbone.children())[:-1])  # Remove FC layer
visual_backbone.eval()


def extract_visual_features(video_path, num_frames=16):
    """Matches the logic in extract_visuals.py"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < num_frames: return None

        # Select 16 frames evenly across the clip (Critical for temporal consistency)
        idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = np.array(frames)[idx]

        # Normalize: (Batch, Channels, Time, Height, Width)
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        video_tensor = video_tensor.unsqueeze(0)

        with torch.no_grad():
            feat = visual_backbone(video_tensor)
            return feat.flatten().numpy()  # Flatten to 512-dim
    except Exception:
        return None


def analyze_video(video_path):
    # A. WHISPER TRANSCRIPTION
    result = whisper_model.transcribe(video_path, fp16=False)
    text = result['text'].strip()
    print(f"TRANSCRIPT: {text}")

    # B. FEATURE EXTRACTION
    audio_feat = extract_audio_features(video_path)  # 692-dim
    visual_feat = extract_visual_features(video_path)  # 512-dim

    # Linguistic: RoBERTa (768) + Liu (7) = 775
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = ling_model(**inputs)
        pooled = getattr(outputs, 'pooler_output', None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        roberta_vec = pooled.numpy().flatten()
    liu_feats = get_liu_handcrafted_features(text)
    text_feat = np.concatenate([roberta_vec, liu_feats])

    # C. FINAL WEIGHTED FUSION (M8)
    if audio_feat is not None and visual_feat is not None:
        with torch.no_grad():
            # Force 2D shape (1, Features) to satisfy torch.cat(dim=1)
            a_t = torch.FloatTensor(audio_feat).reshape(1, -1)
            v_t = torch.FloatTensor(visual_feat).reshape(1, -1)
            l_t = torch.FloatTensor(text_feat).reshape(1, -1)

            output = fusion_model(a_t, v_t, l_t)
            prob = torch.softmax(output, dim=1)
            prediction = torch.argmax(prob, dim=1).item()
            weights = torch.softmax(fusion_model.modality_logits, dim=0)

        # D. OUTPUT RESULTS
        verdict = "DECEPTIVE" if prediction == 1 else "TRUTHFUL"
        print(f"\n--- RESULTS FOR {os.path.basename(video_path)} ---")
        print(f"VERDICT: {verdict} ({prob[0][prediction].item() * 100:.2f}% Confidence)")
        print(f"TRUST: Audio({weights[0]:.2f}) Visual({weights[1]:.2f}) Text({weights[2]:.2f})")
    else:
        print("Error: Could not extract features from video.")

if __name__ == "__main__":
    video_to_test = config.CLIPS_DIR + "test/trial_lie_058.mp4"
    analyze_video(video_to_test)
import torch
import cv2
import numpy as np
import whisper
import os
from transformers import AutoTokenizer
from models.cnn_3d import MADVNet
from models.lie_x_berta import LieXBerta
from models.fusion_model import DeceptionFusion
import imageio_ffmpeg as ffmpeg
import shutil


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


def process_segment(video_tensor, text, models, tokenizer, device):
    v_model, t_model, f_model = models
    encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    ids, mask = encoded['input_ids'].to(device), encoded['attention_mask'].to(device)

    with torch.no_grad():
        # Get independent raw outputs (logits)
        v_logits = v_model(video_tensor.to(device))
        t_logits = t_model(ids, mask)

        # Get fusion features and final output
        v_feat = v_model.get_features(video_tensor.to(device))
        t_feat = t_model.get_features(ids, mask)
        f_logits = f_model(v_feat, t_feat)

        # Convert all to probabilities
        v_probs = torch.softmax(v_logits, dim=1)[0]
        t_probs = torch.softmax(t_logits, dim=1)[0]
        f_probs = torch.softmax(f_logits, dim=1)[0]

        return v_probs, t_probs, f_probs


def run_aggregate_prediction(video_path, segment_duration=5):
    device = torch.device("cpu")
    # Using 'base' instead of 'medium' for faster performance on your i7
    whisper_model = whisper.load_model("base")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load Models
    v_model = MADVNet(num_classes=3).to(device)
    v_model.load_state_dict(torch.load("madv_net_final.pth", map_location=device))
    v_model.eval()
    t_model = LieXBerta(num_classes=3).to(device)
    t_model.load_state_dict(torch.load("lie_x_berta_final.pth", map_location=device))
    t_model.eval()
    f_model = DeceptionFusion(v_dim=128, t_dim=768).to(device)
    f_model.load_state_dict(torch.load("fusion_model_final.pth", map_location=device))
    f_model.eval()
    models = (v_model, t_model, f_model)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Detected video duration: {duration:.2f}s")

    # FIX: If video is shorter than segment_duration, adjust the duration to match the video
    current_segment_len = min(segment_duration, duration)

    # We ensure the loop runs at least once if duration > 0
    loop_range = range(0, max(1, int(duration)), int(current_segment_len))

    all_segment_probs = []

    for start_time in loop_range:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        frames = []

        # Calculate how many frames to skip to get 16 samples across the available time
        skip_rate = max(1, int((fps * current_segment_len) / 16))

        for _ in range(16):
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.resize(frame, (112, 112)) / 255.0)
            # Skip frames to spread the 16 samples across the segment duration
            for _ in range(skip_rate - 1): cap.grab()

        if len(frames) < 16:
            print(f"Skipping segment at {start_time}s: Not enough frames ({len(frames)})")
            continue

        # Transcribe the actual segment
        # Using 'translate' is great if there is Chichewa code-switching
        result = whisper_model.transcribe(video_path, task="translate", verbose=False, fp16=False)
        segment_text = result['text'].lower().strip()

        # Prepare video tensor
        video_tensor = torch.tensor(np.array(frames).transpose(3, 0, 1, 2), dtype=torch.float32).unsqueeze(0)

        # 1. Get independent and fused predictions
        v_p, t_p, f_p = process_segment(video_tensor, segment_text, models, tokenizer, device)

        # 2. Store the fused probability for the final AGGREGATE VERDICT
        all_segment_probs.append(f_p.cpu().numpy())

        # 3. Label mapping
        classes = ['Truth', 'Lie', 'Deception']

        # 4. Determine highest probability class for each brain
        v_res = classes[np.argmax(v_p.cpu().numpy())]
        t_res = classes[np.argmax(t_p.cpu().numpy())]
        f_res = classes[np.argmax(f_p.cpu().numpy())]

        # 5. Detailed Printout for your Thesis Analysis
        print(f"\n--- Segment Analysis ({start_time}s - {start_time + current_segment_len:.1f}s) ---")
        print(f"Transcript: \"{segment_text.strip()}\"")
        print(f"  [Visual Brain (CNN)]:    {v_res.ljust(10)} ({np.max(v_p.cpu().numpy()) * 100:.1f}%)")
        print(f"  [Linguistic (RoBERTa)]: {t_res.ljust(10)} ({np.max(t_p.cpu().numpy()) * 100:.1f}%)")
        print(f"  [MADV-Net Fusion]:      {f_res.ljust(10)} ({np.max(f_p.cpu().numpy()) * 100:.1f}%)")
        print("-" * 30)

    cap.release()

    # FINAL SAFETY CHECK: Ensure we actually found segments
    if not all_segment_probs:
        print("\n" + "!" * 40)
        print("CRITICAL ERROR: No valid segments found to analyze.")
        print("Check if the video file path is correct and accessible.")
        print("!" * 40)
        return

    # Aggregate Results
    avg_probs = np.mean(all_segment_probs, axis=0)
    final_pred = np.argmax(avg_probs)
    classes = ['Truth', 'Lie', 'Deception']

    print("\n" + "=" * 40)
    print(f"AGGREGATE VERDICT: {classes[final_pred].upper()}")
    print(f"OVERALL CONFIDENCE: {avg_probs[final_pred] * 100:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    run_aggregate_prediction("C:/Users/User/Documents/Projects/main-project/data/clips/AN_WILTY_EP15_lie4.mp4")
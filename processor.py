import os
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from preprocessing.video_util import preprocess_video
from extract_linguistic_features import get_liu_handcrafted_features
import imageio_ffmpeg as ffmpeg
import librosa


def ensure_ffmpeg_path():
    """Ensure FFmpeg is available for subprocess audio extraction."""
    try:
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] += os.pathsep + ffmpeg_dir
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
    except Exception:
        pass


def extract_audio_features(video_path):
    """Extract 692-dimensional audio statistics from a video file."""
    ensure_ffmpeg_path()
    tmp_wav = None

    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav = tmp_file.name
        tmp_file.close()

        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ar",
            "22050",
            "-ac",
            "1",
            tmp_wav,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        y, sr = librosa.load(tmp_wav, sr=22050, duration=60)
        if len(y) == 0:
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        all_stats = []
        for feature_set in [mfccs, chroma, mel, contrast, tonnetz]:
            all_stats.append(np.mean(feature_set, axis=1))
            all_stats.append(np.std(feature_set, axis=1))
            all_stats.append(np.max(feature_set, axis=1))
            all_stats.append(np.min(feature_set, axis=1))

        return np.concatenate(all_stats)
    except Exception:
        return None
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


class WeightedFusionBrain(nn.Module):
    def __init__(self, audio_dim=692, visual_dim=512, text_dim=775):
        super(WeightedFusionBrain, self).__init__()
        self.modality_logits = nn.Parameter(torch.tensor([1.5, 1.2, 0.4]))
        self.network = nn.Sequential(
            nn.Linear(audio_dim + visual_dim + text_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2),
        )

    def forward(self, a, v, t):
        weights = torch.softmax(self.modality_logits, dim=0)
        weighted_a = a * weights[0]
        weighted_v = v * weights[1]
        weighted_t = t * weights[2]
        combined = torch.cat((weighted_a, weighted_v, weighted_t), dim=1)
        return self.network(combined)


class DeceptionProcessor:
    def __init__(self, v_weights, t_weights, f_weights):
        self.device = torch.device("cpu")
        ensure_ffmpeg_path()

        try:
            visual_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        except Exception:
            visual_model = r2plus1d_18(pretrained=True)

        self.visual_backbone = torch.nn.Sequential(*list(visual_model.children())[:-1]).to(self.device)
        self.visual_backbone.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.text_model = AutoModel.from_pretrained("roberta-base").to(self.device)
        self.text_model.eval()

        self.fusion_model = WeightedFusionBrain().to(self.device)
        self.fusion_model.load_state_dict(torch.load(f_weights, map_location=self.device))
        self.fusion_model.eval()

    def extract_visual_features(self, video_path):
        video_tensor = preprocess_video(video_path).to(self.device)
        with torch.no_grad():
            features = self.visual_backbone(video_tensor)
            return features.view(features.size(0), -1)

    def extract_text_features(self, transcript_text):
        transcript_text = transcript_text or ""
        inputs = self.tokenizer(
            transcript_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            pooled_output = getattr(outputs, "pooler_output", None)
            if pooled_output is None:
                pooled_output = outputs.last_hidden_state[:, 0, :]

        liu_feats = get_liu_handcrafted_features(transcript_text)
        liu_tensor = torch.tensor(liu_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        return torch.cat((pooled_output, liu_tensor), dim=1)

    def analyze(self, video_path, transcript_text):
        audio_features = extract_audio_features(video_path)
        if audio_features is None:
            raise RuntimeError("audio_feature_extraction_failed")

        audio_tensor = torch.tensor(audio_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        visual_tensor = self.extract_visual_features(video_path)
        text_tensor = self.extract_text_features(transcript_text)

        with torch.no_grad():
            logits = self.fusion_model(audio_tensor, visual_tensor, text_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            modality_weights = torch.softmax(self.fusion_model.modality_logits, dim=0).cpu().tolist()

        classes = ["Truth", "Deception"]
        return {
            "prediction": classes[prediction],
            "confidence": round(probs[0][prediction].item() * 100, 2),
            "modalities": {
                "audio": round(modality_weights[0] * 100, 2),
                "visual": round(modality_weights[1] * 100, 2),
                "text": round(modality_weights[2] * 100, 2),
            },
        }

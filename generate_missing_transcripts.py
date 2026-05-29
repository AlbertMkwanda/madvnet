# Missing Transcript Detection and Generation Script
import pandas as pd
import os
import whisper
import imageio_ffmpeg as ffmpeg
import shutil
import config
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

def setup_ffmpeg():
    """Setup FFmpeg environment for Whisper."""
    actual_ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(actual_ffmpeg_path)
    standard_ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")

    if not os.path.exists(standard_ffmpeg_path):
        shutil.copy(actual_ffmpeg_path, standard_ffmpeg_path)
    
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    os.environ["IMAGEIO_FFMPEG_EXE"] = actual_ffmpeg_path

def load_whisper_model():
    """Load Whisper medium model."""
    print("Loading Whisper MEDIUM model...")
    return whisper.load_model("medium", device="cpu")

def transcribe_video(video_path: str, model) -> str:
    """Transcribe a single video file."""
    try:
        result = model.transcribe(
            video_path,
            fp16=False,
            language='en',
            condition_on_previous_text=False,
            no_speech_threshold=0.2,
            initial_prompt="A conversation from a British comedy game show.",
            temperature=0.0
        )
        clean_text = result['text'].strip().lower()
        return clean_text if clean_text else ""
    except Exception as e:
        print(f"  Error transcribing {os.path.basename(video_path)}: {e}")
        return ""

def find_missing_transcripts(df):
    """Find records with missing or empty transcripts."""
    missing_indices = []
    
    for idx, row in df.iterrows():
        transcript = row['transcript']
        # Check if transcript is missing, NaN, empty string, or None
        if pd.isna(transcript) or (isinstance(transcript, str) and not transcript.strip()):
            missing_indices.append(idx)
    
    return missing_indices

def check_video_exists(file_name: str) -> bool:
    """Check if video file exists."""
    video_path = os.path.normpath(os.path.abspath(
        os.path.join(config.CLIPS_DIR, f"{str(file_name).strip()}.mp4")
    ))
    return os.path.exists(video_path)

def get_video_path(file_name: str) -> str:
    """Get full path to video file."""
    return os.path.normpath(os.path.abspath(
        os.path.join(config.CLIPS_DIR, f"{str(file_name).strip()}.mp4")
    ))

def save_checkpoint(checkpoint_file: str, data: dict):
    """Save progress checkpoint to JSON file."""
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_checkpoint(checkpoint_file: str) -> dict:
    """Load progress checkpoint from JSON file."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return {}
    return {}

def main():
    print("=" * 70)
    print("MISSING TRANSCRIPT DETECTION & GENERATION (with Resume Support)")
    print("=" * 70)

    # Load the deduplicated CSV
    csv_path = config.FINAL_CSV.replace(".csv", "_dedup.csv")
    checkpoint_file = csv_path.replace(".csv", "_generation_checkpoint.json")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        print(f"Please ensure paraphrase_duplicates.py has been run first.")
        return

    print(f"\n📂 Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records")

    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_file)
    processed_indices = set(checkpoint.get('processed_indices', []))
    generation_log = checkpoint.get('generation_log', {})
    
    if processed_indices:
        print(f"\n✓ Resuming from checkpoint: {len(processed_indices)} already processed")
    
    # Find missing transcripts
    print(f"\n🔍 Scanning for missing transcripts...")
    missing_indices = find_missing_transcripts(df)
    
    if not missing_indices:
        print("✓ No missing transcripts found!")
        return

    print(f"⚠️  Found {len(missing_indices)} records with missing transcripts\n")

    # Filter to records with existing video files
    print("📹 Checking which videos exist...")
    missing_with_videos = []
    missing_without_videos = []

    for idx in missing_indices:
        file_name = df.loc[idx, 'file_name']
        if check_video_exists(file_name):
            missing_with_videos.append(idx)
        else:
            missing_without_videos.append(idx)

    print(f"  ✓ {len(missing_with_videos)} records with missing videos available")
    print(f"  ✗ {len(missing_without_videos)} records with missing video files")

    # Filter out already processed
    pending_indices = [idx for idx in missing_with_videos if idx not in processed_indices]
    print(f"  ⏳ {len(pending_indices)} pending transcription")
    
    if not pending_indices:
        print("\nAll videos have been processed. Generating final output...")
        # Continue to final save below
    else:
        # Load Whisper model
        setup_ffmpeg()
        model = load_whisper_model()

        # Generate missing transcripts
        print(f"\n🎙️  Generating missing transcripts...\n")
        
        try:
            for idx in tqdm(pending_indices, desc="Transcribing", unit="video"):
                file_name = df.loc[idx, 'file_name']
                video_path = get_video_path(file_name)

                if not os.path.exists(video_path):
                    generation_log[str(idx)] = {
                        'file_name': file_name,
                        'status': 'VIDEO_NOT_FOUND',
                        'transcript': ''
                    }
                    processed_indices.add(idx)
                    continue

                try:
                    transcript = transcribe_video(video_path, model)
                    
                    if transcript:
                        df.at[idx, 'transcript'] = transcript
                        generation_log[str(idx)] = {
                            'file_name': file_name,
                            'status': 'SUCCESS',
                            'transcript': transcript,
                            'length': len(transcript.split())
                        }
                    else:
                        generation_log[str(idx)] = {
                            'file_name': file_name,
                            'status': 'EMPTY_RESULT',
                            'transcript': ''
                        }

                except Exception as e:
                    generation_log[str(idx)] = {
                        'file_name': file_name,
                        'status': f'ERROR: {str(e)[:50]}',
                        'transcript': ''
                    }

                # Mark as processed
                processed_indices.add(idx)
                
                # Save checkpoint every 5 videos
                if len(processed_indices) % 5 == 0:
                    save_checkpoint(checkpoint_file, {
                        'processed_indices': list(processed_indices),
                        'generation_log': generation_log,
                        'timestamp': datetime.now().isoformat()
                    })
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Script interrupted. Progress saved to checkpoint.")
            save_checkpoint(checkpoint_file, {
                'processed_indices': list(processed_indices),
                'generation_log': generation_log,
                'timestamp': datetime.now().isoformat()
            })
            print(f"Resume with: python generate_missing_transcripts.py")
            return
        except Exception as e:
            print(f"\n\n⚠️  Error occurred: {e}")
            save_checkpoint(checkpoint_file, {
                'processed_indices': list(processed_indices),
                'generation_log': generation_log,
                'timestamp': datetime.now().isoformat()
            })
            print(f"Progress saved. Resume with: python generate_missing_transcripts.py")
            return

    # Save updated CSV
    output_csv = csv_path.replace(".csv", "_complete.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved completed CSV: {output_csv}")

    # Generate report
    report_path = csv_path.replace(".csv", "_generation_report.txt")
    successful = sum(1 for log in generation_log.values() if log.get('status') == 'SUCCESS')
    failed = len(generation_log) - successful
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MISSING TRANSCRIPT GENERATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Missing Transcripts Found: {len(missing_indices)}\n")
        f.write(f"Videos Available: {len(missing_with_videos)}\n")
        f.write(f"Videos Missing: {len(missing_without_videos)}\n")
        f.write(f"Successfully Generated: {successful}\n")
        f.write(f"Generation Failed: {failed}\n")
        f.write(f"Complete Records Now: {len(df) - (len(missing_indices) - successful)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("DETAILED GENERATION LOG:\n")
        f.write("-" * 70 + "\n\n")

        for idx_str, log_info in sorted(generation_log.items(), key=lambda x: int(x[0])):
            idx = int(idx_str)
            f.write(f"Index: {idx}\n")
            f.write(f"File: {log_info['file_name']}\n")
            f.write(f"Status: {log_info['status']}\n")
            if log_info['status'] == 'SUCCESS':
                f.write(f"Word Count: {log_info.get('length', 0)}\n")
                f.write(f"Transcript: {log_info['transcript'][:100]}...\n")
            f.write("-" * 70 + "\n\n")

        # Summary by status
        f.write("\nSUMMARY BY STATUS:\n")
        f.write("-" * 70 + "\n")
        status_counts = {}
        for log_info in generation_log.values():
            status = log_info['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        for status, count in sorted(status_counts.items()):
            f.write(f"{status}: {count}\n")

    print(f"📊 Report saved: {report_path}")

    # Clean up checkpoint if all done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Checkpoint cleaned up")

    # Print summary
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully generated: {successful}")
    print(f"✗ Failed/Empty: {failed}")
    print(f"✗ Missing video files: {len(missing_without_videos)}")
    print("\nFiles created:")
    print(f"  - {output_csv}")
    print(f"  - {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

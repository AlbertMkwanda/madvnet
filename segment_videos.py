import pandas as pd
import subprocess
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
import imageio_ffmpeg as ffmpeg
import config

# ==========================================
# FFMPEG SETUP (same as extraction scripts)
# ==========================================
actual_ffmpeg_path = ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(actual_ffmpeg_path)
standard_ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(standard_ffmpeg_path):
    shutil.copy(actual_ffmpeg_path, standard_ffmpeg_path)
os.environ["PATH"] += os.pathsep + ffmpeg_dir
os.environ["IMAGEIO_FFMPEG_EXE"] = actual_ffmpeg_path

# ==========================================
# CONFIGURATION
# ==========================================
SEGMENT_DURATION_MS = 5000  # 5 seconds in milliseconds
INPUT_CSV = config.FINAL_CSV
OUTPUT_CSV = config.SEGMENTED_DATASET_CSV
VIDEO_INPUT_DIR = config.CLIPS_DIR.replace("\\", "/")
VIDEO_OUTPUT_DIR = config.SEGMENTED_CLIPS_DIR.replace("\\", "/")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def ms_to_seconds(milliseconds):
    """Convert milliseconds to seconds as a float."""
    return milliseconds / 1000.0


def resolve_video_path(video_dir, file_name):
    """
    Resolve the actual video path, handling missing extensions.
    
    Tries:
    1. As-is (file_name already has extension)
    2. With .mp4 extension added
    3. With .mkv extension added
    4. Any video file matching the base name
    
    Returns the full path if found, None otherwise.
    """
    # Normalize video_dir to use backslashes for Windows file operations
    video_dir_windows = video_dir.replace("/", "\\")
    
    # Try as-is first (use backslashes for Windows file checking)
    full_path = os.path.join(video_dir_windows, file_name)
    if os.path.exists(full_path):
        return full_path.replace("\\", "/")
    
    # Try adding common extensions
    for ext in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']:
        full_path = os.path.join(video_dir_windows, f"{file_name}{ext}")
        if os.path.exists(full_path):
            return full_path.replace("\\", "/")
    
    # Try finding any file that starts with the base name
    base_name = Path(file_name).stem
    try:
        for file in os.listdir(video_dir_windows):
            if file.startswith(base_name) and not file.endswith('.tmp'):
                full_path = os.path.join(video_dir_windows, file)
                if os.path.isfile(full_path):
                    return full_path.replace("\\", "/")
    except Exception:
        pass
    
    return None


def get_video_duration(video_path):
    """
    Get video duration in milliseconds by running ffmpeg and parsing output.
    Returns None if unable to determine.
    """
    try:
        import imageio_ffmpeg
        
        # Verify file exists using Windows path format
        windows_path = video_path.replace("/", "\\")
        if not os.path.exists(windows_path):
            logger.warning(f"Video file does not exist: {windows_path}")
            return None
        
        # Get ffmpeg executable from imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Run ffmpeg on the file (convert forward slashes for subprocess)
        ffmpeg_input = video_path.replace("\\", "/")
        
        # Run ffmpeg - it will print to stderr and exit with error (no output file specified)
        # But we can parse the duration from the output
        result = subprocess.run(
            [ffmpeg_exe, '-i', ffmpeg_input],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Parse stderr output for duration (format: Duration: HH:MM:SS.ms)
        output = result.stderr + result.stdout
        import re
        duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.?\d*)', output)
        
        if duration_match:
            hours = int(duration_match.group(1))
            minutes = int(duration_match.group(2))
            seconds = float(duration_match.group(3))
            total_seconds = hours * 3600 + minutes * 60 + seconds
            duration_ms = int(total_seconds * 1000)
            return duration_ms
        
        return None
    except Exception as e:
        logger.warning(f"Failed to get duration for {video_path}: {e}")
        return None


def extract_video_segment(input_path, output_path, start_ms, end_ms):
    """
    Extract a video segment using ffmpeg with -c copy flag (fast, lossless).
    Times remain in milliseconds internally.
    
    Args:
        input_path: Full path to input video
        output_path: Full path to output segment
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert milliseconds to seconds for ffmpeg
        start_sec = ms_to_seconds(start_ms)
        end_sec = ms_to_seconds(end_ms)
        
        # Convert to forward slashes for ffmpeg (cross-platform)
        input_path = input_path.replace("\\", "/")
        output_path = output_path.replace("\\", "/")
        
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-ss', str(start_sec),  # Start time (in seconds for ffmpeg)
            '-to', str(end_sec),  # End time (in seconds for ffmpeg)
            '-i', input_path,
            '-c', 'copy',  # Copy codec (fast, lossless)
            '-avoid_negative_ts', 'make_zero',  # Avoid negative timestamps
            output_path
        ]
        
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )
        
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error extracting segment {output_path}: {e}")
        return False


def process_dataset(input_csv, output_csv, video_input_dir, video_output_dir):
    """
    Main processing function to segment all videos in the dataset.
    """
    # Create output directory if it doesn't exist (use backslashes for Windows)
    os.makedirs(video_output_dir.replace("/", "\\"), exist_ok=True)
    
    # Read the original CSV
    if not os.path.exists(input_csv):
        logger.error(f"Input CSV not found: {input_csv}")
        return False
    
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} rows from {input_csv}")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return False
    
    # Validate required columns
    required_columns = ['YT_Video_ID', 'file_name', 'start_time', 'end_time', 'label', 'transcript']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns. Expected: {required_columns}")
        logger.error(f"Found: {list(df.columns)}")
        return False
    
    # List to store segmented rows
    segmented_rows = []
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        try:
            video_id = row['YT_Video_ID']
            file_name = row['file_name']
            original_start_ms = int(row['start_time'])
            original_end_ms = int(row['end_time'])
            label = row['label']
            transcript = row['transcript']
            
            # Resolve the actual video path (handles missing extensions)
            input_video_path = resolve_video_path(video_input_dir, file_name)
            
            # Check if video exists
            if input_video_path is None:
                logger.warning(f"Video not found: {os.path.join(video_input_dir, file_name)} (tried common extensions)")
                continue
            
            # Get video duration
            video_duration_ms = get_video_duration(input_video_path)
            if video_duration_ms is None:
                logger.warning(f"Could not determine duration for {file_name}, skipping")
                continue
            
            # Calculate actual end time (clamp to video duration)
            actual_end_ms = min(original_end_ms, video_duration_ms)
            
            # Calculate number of 5-second segments
            total_duration_ms = actual_end_ms - original_start_ms
            num_segments = (total_duration_ms + SEGMENT_DURATION_MS - 1) // SEGMENT_DURATION_MS  # Ceiling division
            
            # Extract base filename without extension from the actual found video file
            actual_filename = os.path.basename(input_video_path)
            base_name = Path(actual_filename).stem
            file_extension = Path(actual_filename).suffix
            
            # Create segments
            for seg_idx in range(num_segments):
                segment_start_ms = original_start_ms + (seg_idx * SEGMENT_DURATION_MS)
                segment_end_ms = min(
                    original_start_ms + ((seg_idx + 1) * SEGMENT_DURATION_MS),
                    actual_end_ms
                )
                
                # Create output filename with segment index
                segment_file_name = f"{base_name}_seg{seg_idx + 1}{file_extension}"
                output_video_path = os.path.join(video_output_dir, segment_file_name).replace("\\", "/")
                
                # Extract segment using ffmpeg
                success = extract_video_segment(
                    input_video_path,
                    output_video_path,
                    segment_start_ms,
                    segment_end_ms
                )
                
                if success:
                    # Create row for segmented CSV
                    segmented_row = {
                        'YT_Video_ID': video_id,
                        'file_name': segment_file_name,
                        'start_time': segment_start_ms,
                        'end_time': segment_end_ms,
                        'label': label,
                        'transcript': transcript
                    }
                    segmented_rows.append(segmented_row)
                    logger.info(f"[OK] Created segment: {segment_file_name} ({segment_start_ms}ms - {segment_end_ms}ms)")
                else:
                    logger.error(f"[FAILED] Failed to create segment: {segment_file_name}")
        
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue
    
    # Create output dataframe and save
    if len(segmented_rows) == 0:
        logger.error("No segments were created!")
        return False
    
    output_df = pd.DataFrame(segmented_rows)
    
    try:
        output_df.to_csv(output_csv, index=False)
        logger.info(f"\n[OK] Segmentation complete!")
        logger.info(f"  Original rows: {len(df)}")
        logger.info(f"  Segmented rows: {len(output_df)}")
        logger.info(f"  Output CSV: {output_csv}")
        logger.info(f"  Output videos: {video_output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to write output CSV: {e}")
        return False


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    logger.info("="*70)
    logger.info("VIDEO SEGMENTATION SCRIPT")
    logger.info("="*70)
    logger.info(f"Segment Duration: {SEGMENT_DURATION_MS}ms")
    logger.info(f"Input CSV: {INPUT_CSV}")
    logger.info(f"Output CSV: {OUTPUT_CSV}")
    logger.info(f"Input Video Directory: {VIDEO_INPUT_DIR}")
    logger.info(f"Output Video Directory: {VIDEO_OUTPUT_DIR}")
    logger.info("="*70)
    
    success = process_dataset(INPUT_CSV, OUTPUT_CSV, VIDEO_INPUT_DIR, VIDEO_OUTPUT_DIR)
    
    if success:
        logger.info("\n" + "="*70)
        logger.info("[OK] SEGMENTATION SUCCESSFUL")
        logger.info("="*70)
    else:
        logger.error("\n" + "="*70)
        logger.error("[FAILED] SEGMENTATION FAILED")
        logger.error("="*70)

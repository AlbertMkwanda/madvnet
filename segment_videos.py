import pandas as pd
import subprocess
import os
from pathlib import Path
from tqdm import tqdm
import logging

# ==========================================
# CONFIGURATION
# ==========================================
SEGMENT_DURATION_MS = 5000  # 5 seconds in milliseconds
INPUT_CSV = "C:/Users/User/Documents/Projects/main-project/data/Final_CSV.csv"
OUTPUT_CSV = "C:/Users/User/Documents/Projects/main-project/data/segmented_dataset.csv"
VIDEO_INPUT_DIR = "C:/Users/User/Documents/Projects/main-project/data/clips/"
VIDEO_OUTPUT_DIR = "C:/Users/User/Documents/Projects/main-project/data/segmented_clips/"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation.log'),
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


def get_video_duration(video_path):
    """
    Get video duration in milliseconds using ffprobe.
    Returns None if unable to determine.
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1:novalue=1',
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            duration_seconds = float(result.stdout.strip())
            duration_ms = int(duration_seconds * 1000)
            return duration_ms
        return None
    except Exception as e:
        logger.warning(f"Failed to get duration for {video_path}: {e}")
        return None


def extract_video_segment(input_path, output_path, start_ms, end_ms):
    """
    Extract a video segment using ffmpeg with -c copy flag (fast, lossless).
    
    Args:
        input_path: Full path to input video
        output_path: Full path to output segment
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        start_sec = ms_to_seconds(start_ms)
        end_sec = ms_to_seconds(end_ms)
        duration_sec = end_sec - start_sec
        
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-ss', str(start_sec),  # Start time
            '-to', str(end_sec),  # End time
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
    # Create output directory if it doesn't exist
    os.makedirs(video_output_dir, exist_ok=True)
    
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
            
            # Construct full input path
            input_video_path = os.path.join(video_input_dir, f"{file_name}")
            
            # Check if video exists
            if not os.path.exists(input_video_path):
                logger.warning(f"Video not found: {input_video_path}")
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
            
            # Extract base filename without extension
            base_name = Path(file_name).stem
            file_extension = Path(file_name).suffix
            
            # Create segments
            for seg_idx in range(num_segments):
                segment_start_ms = original_start_ms + (seg_idx * SEGMENT_DURATION_MS)
                segment_end_ms = min(
                    original_start_ms + ((seg_idx + 1) * SEGMENT_DURATION_MS),
                    actual_end_ms
                )
                
                # Create output filename with segment index
                segment_file_name = f"{base_name}_seg{seg_idx + 1}{file_extension}"
                output_video_path = os.path.join(video_output_dir, segment_file_name)
                
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
                    logger.info(f"✓ Created segment: {segment_file_name} ({segment_start_ms}ms - {segment_end_ms}ms)")
                else:
                    logger.error(f"✗ Failed to create segment: {segment_file_name}")
        
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
        logger.info(f"\n✓ Segmentation complete!")
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
        logger.info("✓ SEGMENTATION SUCCESSFUL")
        logger.info("="*70)
    else:
        logger.error("\n" + "="*70)
        logger.error("✗ SEGMENTATION FAILED")
        logger.error("="*70)

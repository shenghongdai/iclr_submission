#!/usr/bin/env python3
"""
Extract frames from all videos in MIntRec directory structure.
Saves one frame per video in the same directory as the video file.
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm


def extract_single_frame(video_path: str, frame_time: str = "00:00:02") -> bool:
    """Extract a single frame from video and save it in the same directory"""
    try:
        # Create output path in same directory as video
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).removesuffix(".mp4")
        output_path = os.path.join(video_dir, f"{video_name}.jpg")
        
        # Check if already exists
        if os.path.exists(output_path):
            return True
        
        # Extract frame using ffmpeg
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-i", video_path,
                "-ss", frame_time,  # Seek to specified time
                "-vframes", "1",    # Extract 1 frame
                "-q:v", "2",        # High quality
                "-y",               # Overwrite output
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and os.path.exists(output_path)
        else:
            print("ffmpeg not found, skipping video frame extraction")
            return False
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return False


def find_all_videos(root_dir: str) -> list[str]:
    """Find all .mp4 files in the directory structure"""
    video_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    return video_files


def main():
    parser = argparse.ArgumentParser(description="Extract frames from MIntRec videos")
    parser.add_argument("--mintrec_dir", 
                       default="vlm_data/MIntRec/data/MIntRec",
                       help="Root directory containing MIntRec videos")
    parser.add_argument("--frame_time", 
                       default="00:00:02",
                       help="Time to extract frame from (HH:MM:SS)")
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg: brew install ffmpeg (on macOS)")
        return
    
    # Find all video files
    print(f"Scanning for videos in {args.mintrec_dir}...")
    video_files = find_all_videos(args.mintrec_dir)
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print("No video files found!")
        return
    
    # Extract frames from each video
    print(f"Extracting frames at time {args.frame_time}...")
    successful = 0
    failed = 0
    skipped = 0
    
    for video_path in tqdm(video_files, desc="Extracting frames"):
        # Check if image already exists
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).removesuffix(".mp4")
        image_path = os.path.join(video_dir, f"{video_name}.jpg")
        
        if os.path.exists(image_path):
            skipped += 1
            continue
            
        if extract_single_frame(video_path, args.frame_time):
            successful += 1
        else:
            failed += 1
    
    print(f"\nExtraction completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Total: {len(video_files)}")
    print(f"  Images saved alongside video files")


if __name__ == "__main__":
    main() 
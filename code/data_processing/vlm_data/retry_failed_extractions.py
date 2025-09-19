#!/usr/bin/env python3
"""
Retry failed video frame extractions with different parameters.
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm


def find_failed_videos(mintrec_dir: str) -> list[str]:
    """Find videos that don't have corresponding extracted frames"""
    failed_videos = []
    
    for root, dirs, files in os.walk(mintrec_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                video_name = file.replace('.mp4', '.jpg')
                image_path = os.path.join(root, video_name)
                
                if not os.path.exists(image_path):
                    failed_videos.append(video_path)
    
    return failed_videos


def extract_frame_with_retry(video_path: str, frame_time: str = "00:00:01") -> bool:
    """Extract a single frame with retry logic"""
    try:
        # Create output path in same directory as video
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).removesuffix(".mp4")
        output_path = os.path.join(video_dir, f"{video_name}.jpg")
        
        # Check if already exists
        if os.path.exists(output_path):
            return True
        
        # Try different extraction strategies
        strategies = [
            # Strategy 1: Try at 1 second
            ["ffmpeg", "-i", video_path, "-ss", frame_time, "-vframes", "1", "-q:v", "2", "-y", output_path],
            # Strategy 2: Try at 0.5 seconds
            ["ffmpeg", "-i", video_path, "-ss", "00:00:00.5", "-vframes", "1", "-q:v", "2", "-y", output_path],
            # Strategy 3: Try first frame
            ["ffmpeg", "-i", video_path, "-vframes", "1", "-q:v", "2", "-y", output_path],
            # Strategy 4: Try with different codec
            ["ffmpeg", "-i", video_path, "-ss", frame_time, "-vframes", "1", "-q:v", "2", "-pix_fmt", "yuv420p", "-y", output_path],
        ]
        
        for i, cmd in enumerate(strategies):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and os.path.exists(output_path):
                    return True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        return False
        
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Retry failed video frame extractions")
    parser.add_argument("--mintrec_dir", 
                       default="vlm_data/MIntRec/data/MIntRec",
                       help="Root directory containing MIntRec videos")
    parser.add_argument("--frame_time", 
                       default="00:00:01",
                       help="Time to extract frame from (HH:MM:SS)")
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg is not installed or not in PATH")
        return
    
    # Find failed videos
    print(f"Scanning for failed extractions in {args.mintrec_dir}...")
    failed_videos = find_failed_videos(args.mintrec_dir)
    print(f"Found {len(failed_videos)} videos without extracted frames")
    
    if not failed_videos:
        print("No failed extractions found!")
        return
    
    # Show some examples of failed videos
    print(f"\nExamples of failed videos:")
    for i, video in enumerate(failed_videos[:5]):
        print(f"  {video}")
    if len(failed_videos) > 5:
        print(f"  ... and {len(failed_videos) - 5} more")
    
    # Retry extraction
    print(f"\nRetrying extraction with multiple strategies...")
    successful = 0
    failed = 0
    
    for video_path in tqdm(failed_videos, desc="Retrying extractions"):
        if extract_frame_with_retry(video_path, args.frame_time):
            successful += 1
        else:
            failed += 1
            print(f"Failed: {video_path}")
    
    print(f"\nRetry Results:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total retried: {len(failed_videos)}")
    
    # Final count
    final_images = len([f for f in os.listdir(args.mintrec_dir) if f.endswith('.jpg')])
    print(f"  Total images now: {final_images}")


if __name__ == "__main__":
    main() 
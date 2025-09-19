#!/usr/bin/env python3
"""
Extract and copy specific COCO and Visual7W images referenced in image_classification.json
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

def extract_image_paths(json_file):
    """Extract all unique image paths from the JSON file."""
    print(f"Reading {json_file}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract image paths and categorize by dataset
    image_paths = defaultdict(set)
    dataset_counts = defaultdict(int)
    
    for item in data:
        if 'image_path' in item:
            image_path = item['image_path']
            dataset = item.get('dataset', 'Unknown')
            
            image_paths[dataset].add(image_path)
            dataset_counts[dataset] += 1
    
    return image_paths, dataset_counts

def copy_images_to_folders(image_paths, dataset_counts):
    """Copy images to separate folders for COCO and Visual7W."""
    
    # Create output directories
    coco_output_dir = Path("coco_images_5k")
    visual7w_output_dir = Path("visual7w_images_5k")
    
    coco_output_dir.mkdir(exist_ok=True)
    visual7w_output_dir.mkdir(exist_ok=True)
    
    print(f"\nCopying COCO images to: {coco_output_dir}")
    print(f"Copying Visual7W images to: {visual7w_output_dir}")
    
    # Copy COCO images
    coco_count = 0
    if 'AOKVQA' in image_paths:
        for image_path in image_paths['AOKVQA']:
            if 'coco' in image_path.lower():
                source_path = Path(image_path)
                if source_path.exists():
                    # Extract filename from path
                    filename = source_path.name
                    dest_path = coco_output_dir / filename
                    shutil.copy2(source_path, dest_path)
                    coco_count += 1
                    if coco_count % 500 == 0:
                        print(f"  Copied {coco_count} COCO images...")
                else:
                    print(f"  Warning: Source file not found: {source_path}")
    
    # Copy Visual7W images
    visual7w_count = 0
    if 'Visual7W' in image_paths:
        for image_path in image_paths['Visual7W']:
            if 'visual7w' in image_path.lower():
                source_path = Path(image_path)
                if source_path.exists():
                    # Extract filename from path
                    filename = source_path.name
                    dest_path = visual7w_output_dir / filename
                    shutil.copy2(source_path, dest_path)
                    visual7w_count += 1
                    if visual7w_count % 500 == 0:
                        print(f"  Copied {visual7w_count} Visual7W images...")
                else:
                    print(f"  Warning: Source file not found: {source_path}")
    
    return coco_count, visual7w_count

def main():
    # Extract from image_classification.json
    image_paths, dataset_counts = extract_image_paths('image_classification.json')
    
    print("\n" + "="*60)
    print("IMAGE EXTRACTION SUMMARY")
    print("="*60)
    
    for dataset, paths in image_paths.items():
        print(f"{dataset}: {len(paths)} unique image files from {dataset_counts[dataset]} conversations")
    
    # Copy images to separate folders
    print("\n" + "="*60)
    print("COPYING IMAGES TO SEPARATE FOLDERS")
    print("="*60)
    
    coco_count, visual7w_count = copy_images_to_folders(image_paths, dataset_counts)
    
    print("\n" + "="*60)
    print("COPY COMPLETED")
    print("="*60)
    print(f"COCO images copied: {coco_count}")
    print(f"Visual7W images copied: {visual7w_count}")
    print(f"COCO folder: coco_images_5k/")
    print(f"Visual7W folder: visual7w_images_5k/")
    
    # Save the list of copied files
    with open('copied_files_summary.txt', 'w') as f:
        f.write("COPIED IMAGES SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"COCO images: {coco_count}\n")
        f.write(f"Visual7W images: {visual7w_count}\n\n")
        
        f.write("COCO image paths:\n")
        if 'AOKVQA' in image_paths:
            for path in sorted(image_paths['AOKVQA']):
                if 'coco' in path.lower():
                    f.write(f"  {path}\n")
        
        f.write("\nVisual7W image paths:\n")
        if 'Visual7W' in image_paths:
            for path in sorted(image_paths['Visual7W']):
                if 'visual7w' in path.lower():
                    f.write(f"  {path}\n")
    
    print(f"\nSummary saved to: copied_files_summary.txt")

if __name__ == "__main__":
    main() 
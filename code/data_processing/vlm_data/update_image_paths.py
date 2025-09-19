#!/usr/bin/env python3
"""
Update image paths in image_classification.json to match new folder structure
"""

import json
import os
from pathlib import Path

def update_image_paths():
    """Update image paths in the JSON file to match new folder structure."""
    
    input_file = 'image_classification.json'
    output_file = 'image_classification_updated.json'
    
    # Define the prefix path
    prefix_path = "./vlm_data/"
    
    print(f"Reading {input_file}...")
    
    # Read the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    
    # Counters for tracking changes
    coco_updated = 0
    visual7w_updated = 0
    mintrec_updated = 0
    unchanged = 0
    
    # Update paths
    for item in data:
        if 'image_path' in item:
            old_path = item['image_path']
            
            # Update COCO paths
            if 'aokvqa/datasets/coco/train2017/' in old_path:
                filename = old_path.split('/')[-1]  # Extract filename
                new_path = f"{prefix_path}coco_images_5k/{filename}"
                item['image_path'] = new_path
                coco_updated += 1
                
            # Update Visual7W paths
            elif 'visual7w-toolkit/images/' in old_path:
                filename = old_path.split('/')[-1]  # Extract filename
                new_path = f"{prefix_path}visual7w_images_5k/{filename}"
                item['image_path'] = new_path
                visual7w_updated += 1
                
            # Update MIntRec paths
            elif 'MIntRec/data/MIntRec/' in old_path:
                # Extract the path after MIntRec/data/MIntRec/
                # Example: MIntRec/data/MIntRec/S04/E03/485.mp4 -> S04/E03/485.mp4
                path_parts = old_path.split('MIntRec/data/MIntRec/')
                if len(path_parts) > 1:
                    relative_path = path_parts[1]
                    new_path = f"{prefix_path}MIntRec/{relative_path}"
                    item['image_path'] = new_path
                    mintrec_updated += 1
                else:
                    unchanged += 1
                
            else:
                unchanged += 1
                # Keep original path for any other cases
    
    print(f"\nPath update summary:")
    print(f"  COCO paths updated: {coco_updated}")
    print(f"  Visual7W paths updated: {visual7w_updated}")
    print(f"  MIntRec paths updated: {mintrec_updated}")
    print(f"  Unchanged paths: {unchanged}")
    
    # Save the updated JSON
    print(f"\nSaving updated JSON to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully updated image paths!")
    print(f"📁 Original file: {input_file}")
    print(f"📁 Updated file: {output_file}")
    print(f"📍 Prefix path: {prefix_path}")
    
    # Verify some examples
    print(f"\n📋 Example path updates:")
    coco_examples = 0
    visual7w_examples = 0
    mintrec_examples = 0
    
    for item in data[:15]:  # Show first 15 examples
        if 'image_path' in item:
            path = item['image_path']
            if 'coco_images_5k/' in path and coco_examples < 3:
                print(f"  COCO: {path}")
                coco_examples += 1
            elif 'visual7w_images_5k/' in path and visual7w_examples < 3:
                print(f"  Visual7W: {path}")
                visual7w_examples += 1
            elif 'MIntRec/' in path and mintrec_examples < 3:
                print(f"  MIntRec: {path}")
                mintrec_examples += 1
    
    return output_file

def main():
    print("="*60)
    print("UPDATING IMAGE PATHS IN JSON")
    print("="*60)
    
    try:
        output_file = update_image_paths()
        
        print("\n" + "="*60)
        print("UPDATE COMPLETED")
        print("="*60)
        print(f"✅ Updated JSON saved as: {output_file}")
        print(f"📝 You can now use this file with your uploaded image folders")
        print(f" Upload coco_images_5k/ and visual7w_images_5k/ to your provider")
        print(f"📍 All paths now include the local prefix")
        print(f"🎥 MIntRec video paths updated to match your local structure")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 
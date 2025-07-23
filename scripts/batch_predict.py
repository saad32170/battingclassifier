#!/usr/bin/env python3
"""
Batch Cricket Shot Predictor
Usage: python batch_predict.py folder_path
"""

import sys
import os
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ensemble_predictor import EnsemblePredictor

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_predict.py <folder_path>")
        print("Example: python batch_predict.py drive/")
        return
    
    folder_path = sys.argv[1]
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"No image files found in '{folder_path}'")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    try:
        # Load ensemble predictor
        predictor = EnsemblePredictor()
        
        # Process each image
        results = []
        for i, image_path in enumerate(image_files[:10]):  # Limit to first 10 images
            print(f"\n[{i+1}/{min(len(image_files), 10)}] Processing: {os.path.basename(image_path)}")
            
            shot_type, confidence = predictor.predict_single_image(image_path)
            
            if shot_type:
                results.append({
                    'image': os.path.basename(image_path),
                    'prediction': shot_type,
                    'confidence': confidence
                })
                print(f"   → {shot_type.upper()} ({confidence:.1%})")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"Processed {len(results)} images")
        
        # Count predictions
        shot_counts = {}
        for result in results:
            shot = result['prediction']
            shot_counts[shot] = shot_counts.get(shot, 0) + 1
        
        for shot, count in shot_counts.items():
            print(f"  {shot}: {count} images")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Cricket Shot Predictor - Main Entry Point
Achieved 97.14% accuracy with ensemble learning

Usage:
    python main.py --image path/to/image.jpg
    python main.py --interactive
    python main.py --batch folder_path
"""

import sys
import os
from ensemble_predictor import EnsemblePredictor

def main():
    if len(sys.argv) < 2:
        print("🏏 Cricket Shot Predictor (97.14% Accuracy)")
        print("=" * 50)
        print("Usage:")
        print("  python main.py --image <image_path>     # Single image prediction")
        print("  python main.py --interactive            # Interactive mode")
        print("  python main.py --batch <folder_path>    # Batch prediction")
        print()
        print("Examples:")
        print("  python main.py --image drive/drives1.png")
        print("  python main.py --interactive")
        print("  python main.py --batch drive/")
        return
    
    command = sys.argv[1]
    
    try:
        predictor = EnsemblePredictor()
        
        if command == "--image":
            if len(sys.argv) != 3:
                print("Usage: python main.py --image <image_path>")
                return
            
            image_path = sys.argv[2]
            if not os.path.exists(image_path):
                print(f"Error: Image file '{image_path}' not found!")
                return
            
            shot_type, confidence = predictor.predict_single_image(image_path)
            if shot_type:
                print(f"\n🎯 RESULT: {shot_type.upper()} ({confidence:.1%} confidence)")
            else:
                print("❌ Prediction failed")
        
        elif command == "--interactive":
            predictor.interactive_mode()
        
        elif command == "--batch":
            if len(sys.argv) != 3:
                print("Usage: python main.py --batch <folder_path>")
                return
            
            folder_path = sys.argv[2]
            if not os.path.exists(folder_path):
                print(f"Error: Folder '{folder_path}' not found!")
                return
            
            # Simple batch processing
            import glob
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if not image_files:
                print(f"No image files found in '{folder_path}'")
                return
            
            print(f"Found {len(image_files)} images. Processing first 5...")
            
            results = []
            for i, image_path in enumerate(image_files[:5]):
                print(f"\n[{i+1}/5] {os.path.basename(image_path)}")
                shot_type, confidence = predictor.predict_single_image(image_path)
                if shot_type:
                    results.append((shot_type, confidence))
                    print(f"   → {shot_type.upper()} ({confidence:.1%})")
            
            # Summary
            if results:
                print(f"\n📊 SUMMARY:")
                shot_counts = {}
                for shot, conf in results:
                    shot_counts[shot] = shot_counts.get(shot, 0) + 1
                
                for shot, count in shot_counts.items():
                    print(f"  {shot}: {count} images")
        
        else:
            print(f"Unknown command: {command}")
            print("Use --image, --interactive, or --batch")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
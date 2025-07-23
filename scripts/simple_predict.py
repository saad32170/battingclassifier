#!/usr/bin/env python3
"""
Simple Cricket Shot Predictor
Usage: python simple_predict.py image_path
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ensemble_predictor import EnsemblePredictor

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_predict.py <image_path>")
        print("Example: python simple_predict.py drive/drives1.png")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    try:
        # Load ensemble and predict
        predictor = EnsemblePredictor()
        shot_type, confidence = predictor.predict_single_image(image_path)
        
        if shot_type:
            print(f"\n🎯 RESULT: {shot_type.upper()} ({confidence:.1%} confidence)")
        else:
            print("❌ Prediction failed")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
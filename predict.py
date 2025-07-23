import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import argparse

class CricketShotPredictor:
    def __init__(self, model_path, img_size=(224, 224)):
        """
        Cricket Shot Predictor - Use trained model for predictions
        
        Args:
            model_path: Path to the trained model file (.h5)
            img_size: Input image size expected by the model
        """
        self.model_path = model_path
        self.img_size = img_size
        self.class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        self.model = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.img_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def predict_single_image(self, image_path, show_image=True):
        """Predict shot type for a single image"""
        # Preprocess image
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        
        # Make prediction
        prediction = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Get class name
        predicted_shot = self.class_names[predicted_class]
        
        # Print results
        print(f"\nPrediction for: {os.path.basename(image_path)}")
        print(f"Predicted shot: {predicted_shot}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("\nAll probabilities:")
        for i, (class_name, prob) in enumerate(zip(self.class_names, prediction[0])):
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        # Show image if requested
        if show_image:
            self.show_prediction(image_path, predicted_shot, confidence)
        
        return predicted_shot, confidence, prediction[0]
    
    def predict_batch(self, image_folder, max_images=10):
        """Predict shot types for multiple images in a folder"""
        if not os.path.exists(image_folder):
            print(f"Folder {image_folder} does not exist")
            return
        
        # Get image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No image files found in {image_folder}")
            return
        
        # Limit number of images
        image_files = image_files[:max_images]
        
        print(f"Predicting {len(image_files)} images from {image_folder}")
        print("=" * 60)
        
        results = []
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(image_folder, img_file)
            print(f"\n{i+1}/{len(image_files)}: {img_file}")
            
            result = self.predict_single_image(img_path, show_image=False)
            if result:
                predicted_shot, confidence, probabilities = result
                results.append({
                    'file': img_file,
                    'predicted': predicted_shot,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
        
        # Summary
        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        
        # Count predictions
        shot_counts = {}
        for result in results:
            shot = result['predicted']
            shot_counts[shot] = shot_counts.get(shot, 0) + 1
        
        for shot, count in shot_counts.items():
            print(f"{shot}: {count} images")
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nAverage confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        
        return results
    
    def show_prediction(self, image_path, predicted_shot, confidence):
        """Display image with prediction"""
        try:
            # Load and display image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f'Predicted: {predicted_shot}\nConfidence: {confidence:.4f} ({confidence*100:.2f}%)')
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def interactive_predictor(self):
        """Interactive mode for predicting images"""
        print("Interactive Cricket Shot Predictor")
        print("=" * 40)
        print("Commands:")
        print("  'quit' or 'exit' - Exit the program")
        print("  'help' - Show this help")
        print("  <image_path> - Predict shot type for an image")
        print()
        
        while True:
            try:
                user_input = input("Enter image path (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("Commands:")
                    print("  'quit' or 'exit' - Exit the program")
                    print("  'help' - Show this help")
                    print("  <image_path> - Predict shot type for an image")
                    print()
                elif user_input:
                    if os.path.exists(user_input):
                        self.predict_single_image(user_input)
                    else:
                        print(f"File not found: {user_input}")
                else:
                    print("Please enter an image path or command")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cricket Shot Predictor')
    parser.add_argument('--model', type=str, default='quick_fix_classifier.h5',
                       help='Path to trained model file')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--folder', type=str, help='Path to folder with images for batch prediction')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--max-images', type=int, default=10, help='Maximum images for batch prediction')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CricketShotPredictor(args.model)
    
    # Load model
    if not predictor.load_model():
        print("Failed to load model. Please check the model path.")
        return
    
    # Run appropriate mode
    if args.interactive:
        predictor.interactive_predictor()
    elif args.image:
        predictor.predict_single_image(args.image)
    elif args.folder:
        predictor.predict_batch(args.folder, args.max_images)
    else:
        # Default: interactive mode
        print("No specific mode specified. Running in interactive mode.")
        print("You can also use:")
        print("  python predict.py --image path/to/image.jpg")
        print("  python predict.py --folder path/to/images/")
        print("  python predict.py --interactive")
        print()
        predictor.interactive_predictor()

if __name__ == "__main__":
    main() 
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    def __init__(self, models_dir='saved_models'):
        self.models_dir = models_dir
        self.class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        self.models = []
        self.model_names = []
        self.load_models()
    
    def load_models(self):
        """Load all ensemble models"""
        print("Loading ensemble models...")
        
        # Model 1: Custom CNN
        try:
            model1 = load_model(f'{self.models_dir}/ensemble_model_1.keras')
            self.models.append(model1)
            self.model_names.append('Custom CNN')
            print("✓ Loaded Custom CNN")
        except Exception as e:
            print(f"✗ Failed to load Custom CNN: {e}")
        
        # Model 2: ResNet50V2
        try:
            model2 = load_model(f'{self.models_dir}/ensemble_model_2.keras')
            self.models.append(model2)
            self.model_names.append('ResNet50V2')
            print("✓ Loaded ResNet50V2")
        except Exception as e:
            print(f"✗ Failed to load ResNet50V2: {e}")
        
        # Model 3: Advanced CNN
        try:
            model3 = load_model(f'{self.models_dir}/best_advanced_model.h5')
            self.models.append(model3)
            self.model_names.append('Advanced CNN')
            print("✓ Loaded Advanced CNN")
        except Exception as e:
            print(f"✗ Failed to load Advanced CNN: {e}")
        
        print(f"\nLoaded {len(self.models)} models for ensemble")
        if len(self.models) == 0:
            raise Exception("No models loaded!")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    
    def predict_ensemble(self, image_path, method='weighted_averaging'):
        """Make ensemble prediction"""
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Get predictions from all models
        all_predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(img, verbose=0)  # verbose=0 to suppress output
            all_predictions.append(pred[0])  # Remove batch dimension
            print(f"{self.model_names[i]}: {self.class_names[np.argmax(pred[0])]} ({np.max(pred[0]):.3f})")
        
        all_predictions = np.array(all_predictions)
        
        # Apply ensemble method
        if method == 'voting':
            # Hard voting: majority vote
            pred_classes = np.argmax(all_predictions, axis=1)
            unique, counts = np.unique(pred_classes, return_counts=True)
            final_pred = unique[np.argmax(counts)]
            final_confidence = np.max(counts) / len(self.models)
            
        elif method == 'averaging':
            # Soft voting: average probabilities
            avg_predictions = np.mean(all_predictions, axis=0)
            final_pred = np.argmax(avg_predictions)
            final_confidence = np.max(avg_predictions)
            
        elif method == 'weighted_averaging':
            # Weighted average based on individual model performance
            weights = [0.3, 0.4, 0.3]  # Based on individual accuracies
            weighted_predictions = np.zeros_like(all_predictions[0])
            
            for i, (pred, weight) in enumerate(zip(all_predictions, weights)):
                weighted_predictions += weight * pred
            
            final_pred = np.argmax(weighted_predictions)
            final_confidence = np.max(weighted_predictions)
        
        return self.class_names[final_pred], final_confidence, all_predictions
    
    def predict_single_image(self, image_path, method='weighted_averaging'):
        """Predict shot type for a single image"""
        print(f"\n🎯 Predicting: {image_path}")
        print("=" * 50)
        
        try:
            shot_type, confidence, all_predictions = self.predict_ensemble(image_path, method)
            
            print(f"\n🏆 ENSEMBLE RESULT ({method.upper()}):")
            print(f"Predicted Shot: {shot_type}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # Show individual model predictions
            print(f"\n📊 Individual Model Predictions:")
            for i, pred in enumerate(all_predictions):
                pred_class = self.class_names[np.argmax(pred)]
                pred_conf = np.max(pred)
                print(f"  {self.model_names[i]}: {pred_class} ({pred_conf:.3f})")
            
            return shot_type, confidence
            
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None, None
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("🏏 Cricket Shot Ensemble Predictor")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Cricket Shot Ensemble Predictor')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--method', type=str, default='weighted_averaging', 
                       choices=['voting', 'averaging', 'weighted_averaging'],
                       help='Ensemble method to use')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize ensemble predictor
    try:
        predictor = EnsemblePredictor()
    except Exception as e:
        print(f"Failed to initialize ensemble predictor: {e}")
        return
    
    # Run appropriate mode
    if args.interactive:
        predictor.interactive_mode()
    elif args.image:
        predictor.predict_single_image(args.image, args.method)
    else:
        # Default: interactive mode
        print("No specific mode specified. Running in interactive mode.")
        print("You can also use:")
        print("  python ensemble_predictor.py --image path/to/image.jpg")
        print("  python ensemble_predictor.py --interactive")
        print()
        predictor.interactive_mode()

if __name__ == "__main__":
    main() 
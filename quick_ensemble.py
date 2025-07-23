import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import load_model
import cv2
import warnings
warnings.filterwarnings('ignore')

class QuickEnsemble:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        self.num_classes = len(self.class_names)
        self.models = []
        
    def load_existing_models(self):
        """Load the models that were already trained"""
        print("Loading existing ensemble models...")
        
        # Try to load Model 1 (Custom CNN)
        try:
            model1 = load_model('saved_models/ensemble_model_1.keras')
            self.models.append(('Custom CNN', model1))
            print("✓ Loaded Model 1 (Custom CNN)")
        except:
            print("✗ Model 1 not found")
        
        # Try to load Model 2 (ResNet50V2)
        try:
            model2 = load_model('saved_models/ensemble_model_2.keras')
            self.models.append(('ResNet50V2', model2))
            print("✓ Loaded Model 2 (ResNet50V2)")
        except:
            print("✗ Model 2 not found")
        
        # Try to load the advanced model
        try:
            advanced_model = load_model('saved_models/best_advanced_model.h5')
            self.models.append(('Advanced CNN', advanced_model))
            print("✓ Loaded Advanced Model")
        except:
            print("✗ Advanced model not found")
        
        print(f"Loaded {len(self.models)} models for ensemble")
        return len(self.models) > 0
    
    def load_test_data(self):
        """Load test data for evaluation"""
        print("Loading test data...")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            img = img / 255.0
                            
                            images.append(img)
                            labels.append(class_idx)
                    except Exception as e:
                        continue
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        
        # Split into train/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels, test_size=0.2, 
            stratify=self.labels, random_state=42
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"Test data: {len(X_test)} images")
        return True
    
    def evaluate_individual_models(self):
        """Evaluate each model individually"""
        print("\n" + "="*50)
        print("INDIVIDUAL MODEL EVALUATION")
        print("="*50)
        
        individual_accuracies = []
        
        for i, (model_name, model) in enumerate(self.models):
            print(f"\n--- {model_name} ---")
            
            # Get predictions
            predictions = model.predict(self.X_test)
            y_pred = np.argmax(predictions, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == self.y_test)
            individual_accuracies.append(accuracy)
            
            print(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=self.class_names))
        
        return individual_accuracies
    
    def ensemble_predict(self, X, method='voting'):
        """Make ensemble predictions using different methods"""
        
        if len(self.models) == 0:
            print("No models loaded!")
            return None
        
        # Get predictions from all models
        all_predictions = []
        for model_name, model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples, n_classes)
        
        if method == 'voting':
            # Hard voting: majority vote
            pred_classes = np.argmax(all_predictions, axis=2)  # Shape: (n_models, n_samples)
            ensemble_pred = []
            for i in range(pred_classes.shape[1]):
                votes = pred_classes[:, i]
                # Get most common prediction
                unique, counts = np.unique(votes, return_counts=True)
                ensemble_pred.append(unique[np.argmax(counts)])
            
            return np.array(ensemble_pred)
        
        elif method == 'averaging':
            # Soft voting: average probabilities
            avg_predictions = np.mean(all_predictions, axis=0)  # Shape: (n_samples, n_classes)
            return np.argmax(avg_predictions, axis=1)
        
        elif method == 'weighted_averaging':
            # Weighted average based on individual model performance
            weights = [0.3, 0.4, 0.3]  # Adjust based on model performance
            weighted_predictions = np.zeros_like(all_predictions[0])
            
            for i, (pred, weight) in enumerate(zip(all_predictions, weights)):
                weighted_predictions += weight * pred
            
            return np.argmax(weighted_predictions, axis=1)
    
    def evaluate_ensemble(self):
        """Evaluate the ensemble model"""
        print("\n" + "="*50)
        print("ENSEMBLE EVALUATION")
        print("="*50)
        
        # Evaluate ensemble methods
        ensemble_methods = ['voting', 'averaging', 'weighted_averaging']
        
        best_accuracy = 0
        best_method = None
        
        for method in ensemble_methods:
            print(f"\n--- {method.upper()} ENSEMBLE ---")
            
            # Get ensemble predictions
            y_pred = self.ensemble_predict(self.X_test, method=method)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == self.y_test)
            print(f"Ensemble Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=self.class_names))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title(f'Ensemble Confusion Matrix ({method})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
        
        print(f"\n🏆 BEST ENSEMBLE METHOD: {best_method.upper()}")
        print(f"🏆 BEST ACCURACY: {best_accuracy:.4f}")
        
        return best_accuracy, best_method

def main():
    """Main function to evaluate the ensemble"""
    
    # Initialize ensemble
    ensemble = QuickEnsemble(
        data_dir='.',
        img_size=(224, 224),
        batch_size=32
    )
    
    # Load existing models
    if not ensemble.load_existing_models():
        print("No models found! Please train the ensemble first.")
        return
    
    # Load test data
    ensemble.load_test_data()
    
    # Evaluate individual models
    individual_accuracies = ensemble.evaluate_individual_models()
    
    # Evaluate ensemble
    best_accuracy, best_method = ensemble.evaluate_ensemble()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Individual Models: {[f'{acc:.3f}' for acc in individual_accuracies]}")
    print(f"Ensemble ({best_method}): {best_accuracy:.4f}")
    
    if best_accuracy >= 0.95:
        print("🎉 CONGRATULATIONS! You've achieved 95%+ accuracy!")
    else:
        print(f"📈 Close to 95%! Need {0.95 - best_accuracy:.3f} more improvement")

if __name__ == "__main__":
    main() 
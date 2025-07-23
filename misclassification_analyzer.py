import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class MisclassificationAnalyzer:
    def __init__(self, model_path, data_dir, img_size=(224, 224)):
        self.model_path = model_path
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        self.model = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def load_test_data(self):
        """Load test data for analysis"""
        print("Loading test data...")
        
        images = []
        labels = []
        image_paths = []
        
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
                            image_paths.append(img_path)
                    except Exception as e:
                        continue
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.image_paths = image_paths
        
        print(f"Loaded {len(self.images)} images")
        return True
    
    def analyze_misclassifications(self):
        """Analyze misclassified images in detail"""
        if self.model is None:
            print("Model not loaded!")
            return
        
        print("Analyzing misclassifications...")
        
        # Get predictions
        predictions = self.model.predict(self.images)
        y_pred = np.argmax(predictions, axis=1)
        
        # Find misclassified images
        misclassified_indices = np.where(y_pred != self.labels)[0]
        correct_indices = np.where(y_pred == self.labels)[0]
        
        print(f"\nTotal images: {len(self.images)}")
        print(f"Correctly classified: {len(correct_indices)} ({len(correct_indices)/len(self.images)*100:.2f}%)")
        print(f"Misclassified: {len(misclassified_indices)} ({len(misclassified_indices)/len(self.images)*100:.2f}%)")
        
        # Analyze confusion patterns
        self.analyze_confusion_patterns(y_pred)
        
        # Analyze confidence levels
        self.analyze_confidence_levels(predictions, y_pred)
        
        # Show examples of misclassified images
        self.show_misclassified_examples(misclassified_indices, y_pred, predictions)
        
        # Analyze class-specific issues
        self.analyze_class_specific_issues(y_pred)
        
        return misclassified_indices, y_pred, predictions
    
    def analyze_confusion_patterns(self, y_pred):
        """Analyze which classes are most confused with each other"""
        print("\n=== CONFUSION PATTERN ANALYSIS ===")
        
        cm = confusion_matrix(self.labels, y_pred)
        
        # Find most common misclassifications
        misclassifications = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i][j] > 0:
                    misclassifications.append({
                        'actual': self.class_names[i],
                        'predicted': self.class_names[j],
                        'count': cm[i][j]
                    })
        
        # Sort by count
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        
        print("\nMost Common Misclassifications:")
        for i, mis in enumerate(misclassifications[:10]):  # Top 10
            print(f"{i+1}. {mis['actual']} → {mis['predicted']}: {mis['count']} images")
    
    def analyze_confidence_levels(self, predictions, y_pred):
        """Analyze confidence levels for correct vs incorrect predictions"""
        print("\n=== CONFIDENCE ANALYSIS ===")
        
        # Get confidence levels
        max_confidences = np.max(predictions, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = y_pred == self.labels
        incorrect_mask = y_pred != self.labels
        
        correct_confidences = max_confidences[correct_mask]
        incorrect_confidences = max_confidences[incorrect_mask]
        
        print(f"Correct predictions - Avg confidence: {correct_confidences.mean():.3f}")
        print(f"Incorrect predictions - Avg confidence: {incorrect_confidences.mean():.3f}")
        
        # Find low-confidence correct predictions and high-confidence incorrect predictions
        low_conf_correct = np.sum(correct_confidences < 0.7)
        high_conf_incorrect = np.sum(incorrect_confidences > 0.8)
        
        print(f"Correct predictions with low confidence (<0.7): {low_conf_correct}")
        print(f"Incorrect predictions with high confidence (>0.8): {high_conf_incorrect}")
        
        # Plot confidence distributions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot([correct_confidences, incorrect_confidences], 
                   labels=['Correct', 'Incorrect'])
        plt.ylabel('Confidence')
        plt.title('Confidence Box Plot')
        
        plt.tight_layout()
        plt.show()
    
    def show_misclassified_examples(self, misclassified_indices, y_pred, predictions):
        """Show examples of misclassified images"""
        print("\n=== MISCLASSIFIED IMAGE EXAMPLES ===")
        
        if len(misclassified_indices) == 0:
            print("No misclassified images found!")
            return
        
        # Show first 8 misclassified images
        num_examples = min(8, len(misclassified_indices))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(num_examples):
            idx = misclassified_indices[i]
            actual_class = self.class_names[self.labels[idx]]
            predicted_class = self.class_names[y_pred[idx]]
            confidence = np.max(predictions[idx])
            
            # Load and display image
            img = self.images[idx]
            axes[i].imshow(img)
            axes[i].set_title(f'Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {confidence:.3f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show class-specific examples
        self.show_class_specific_examples(misclassified_indices, y_pred, predictions)
    
    def show_class_specific_examples(self, misclassified_indices, y_pred, predictions):
        """Show examples for each class that was misclassified"""
        print("\n=== CLASS-SPECIFIC MISCLASSIFICATION EXAMPLES ===")
        
        for actual_class_idx, actual_class_name in enumerate(self.class_names):
            # Find misclassified images for this class
            class_misclassified = []
            for idx in misclassified_indices:
                if self.labels[idx] == actual_class_idx:
                    class_misclassified.append(idx)
            
            if len(class_misclassified) > 0:
                print(f"\n{actual_class_name.upper()} misclassified as:")
                
                # Group by predicted class
                predicted_counts = {}
                for idx in class_misclassified:
                    predicted_class = self.class_names[y_pred[idx]]
                    if predicted_class not in predicted_counts:
                        predicted_counts[predicted_class] = []
                    predicted_counts[predicted_class].append(idx)
                
                # Show top misclassifications
                for predicted_class, indices in predicted_counts.items():
                    print(f"  → {predicted_class}: {len(indices)} images")
                    
                    # Show example
                    if len(indices) > 0:
                        example_idx = indices[0]
                        confidence = np.max(predictions[example_idx])
                        print(f"    Example confidence: {confidence:.3f}")
    
    def analyze_class_specific_issues(self, y_pred):
        """Analyze issues specific to each class"""
        print("\n=== CLASS-SPECIFIC ANALYSIS ===")
        
        for class_idx, class_name in enumerate(self.class_names):
            # Find all images of this class
            class_mask = self.labels == class_idx
            class_total = np.sum(class_mask)
            
            # Find correctly classified images of this class
            class_correct = np.sum((self.labels == class_idx) & (y_pred == class_idx))
            class_accuracy = class_correct / class_total
            
            print(f"{class_name}: {class_correct}/{class_total} correct ({class_accuracy:.3f})")
            
            # Find what this class is most confused with
            class_predictions = y_pred[class_mask]
            unique, counts = np.unique(class_predictions, return_counts=True)
            
            for pred_class_idx, count in zip(unique, counts):
                if pred_class_idx != class_idx:
                    pred_class_name = self.class_names[pred_class_idx]
                    percentage = count / class_total
                    print(f"  → {percentage:.1%} misclassified as {pred_class_name}")

def main():
    """Main function to analyze misclassifications"""
    
    # Initialize analyzer
    analyzer = MisclassificationAnalyzer(
        model_path='best_advanced_model.keras',  # Use the best model
        data_dir='.',
        img_size=(224, 224)
    )
    
    # Load model and data
    if not analyzer.load_model():
        return
    
    if not analyzer.load_test_data():
        return
    
    # Analyze misclassifications
    misclassified_indices, y_pred, predictions = analyzer.analyze_misclassifications()
    
    print("\n" + "="*50)
    print("MISCLASSIFICATION ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main() 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
import cv2
import warnings
warnings.filterwarnings('ignore')

class AdvancedCricketShotClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        self.num_classes = len(self.class_names)
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with detailed diagnostics"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        class_counts = {class_name: 0 for class_name in self.class_names}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            print(f"Processing {class_name} images...")
            class_images = 0
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            img = img / 255.0  # Normalize to [0, 1]
                            
                            # Check for NaN or infinite values
                            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                                print(f"Warning: NaN or Inf found in {img_path}")
                                continue
                                
                            images.append(img)
                            labels.append(class_idx)
                            class_counts[class_name] += 1
                            class_images += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
            
            print(f"  Loaded {class_images} images for {class_name}")
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        
        print(f"\nTotal loaded: {len(self.images)} images")
        print(f"Class distribution: {class_counts}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        # Check for data issues
        print(f"\nData shape: {self.images.shape}")
        print(f"Image value range: [{self.images.min():.3f}, {self.images.max():.3f}]")
        print(f"Image mean: {self.images.mean():.3f}")
        print(f"Image std: {self.images.std():.3f}")
        
        return self.images, self.labels
    
    def create_data_generators(self, test_size=0.2, val_size=0.2):
        """Create train, validation, and test splits with balanced augmentation"""
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels, test_size=test_size, 
            stratify=self.labels, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, 
            stratify=y_temp, random_state=42
        )
        
        # Balanced data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=15,  # Slightly more than diagnostic
            width_shift_range=0.15,  # Balanced augmentation
            height_shift_range=0.15,
            shear_range=0.1,  # Add slight shear
            zoom_range=0.15,  # Add slight zoom
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # No augmentation for validation/test
        val_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, self.num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # Create generators
        self.train_generator = train_datagen.flow(
            X_train, y_train_cat, batch_size=self.batch_size
        )
        
        self.val_generator = val_datagen.flow(
            X_val, y_val_cat, batch_size=self.batch_size
        )
        
        self.test_generator = test_datagen.flow(
            X_test, y_test_cat, batch_size=self.batch_size
        )
        
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_cat = y_test_cat
        
        print(f"\nData splits:")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Check class distribution in splits
        print(f"Train class distribution: {np.bincount(y_train)}")
        print(f"Val class distribution: {np.bincount(y_val)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def build_advanced_model(self, model_type='hybrid'):
        """Build an advanced model with transfer learning"""
        
        if model_type == 'resnet':
            # Transfer learning with ResNet50V2
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
            # Freeze base model initially
            base_model.trainable = False
            
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),  # Add batch normalization
                layers.Dropout(0.4),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
        else:
            # Enhanced custom CNN architecture
            self.model = models.Sequential([
                # First conv block
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Second conv block
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Third conv block
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Fourth conv block
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Flatten and dense layers
                layers.Flatten(),
                layers.Dropout(0.4),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile model with optimized settings
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Advanced Model Summary:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=40, patience=20):
        """Train the model with advanced callbacks"""
        
        # Advanced callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        # Model checkpoint to save best model
        checkpoint = callbacks.ModelCheckpoint(
            'best_advanced_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Custom callback to monitor predictions
        class PredictionMonitor(callbacks.Callback):
            def __init__(self, val_data, class_names):
                super().__init__()
                self.val_data = val_data
                self.class_names = class_names
                
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:  # Check every 5 epochs
                    predictions = self.model.predict(self.val_data[0][:100])
                    pred_classes = np.argmax(predictions, axis=1)
                    unique, counts = np.unique(pred_classes, return_counts=True)
                    print(f"Epoch {epoch}: Prediction distribution: {dict(zip([self.class_names[i] for i in unique], counts))}")
        
        prediction_monitor = PredictionMonitor((self.X_test, self.y_test_cat), self.class_names)
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=[early_stopping, reduce_lr, checkpoint, prediction_monitor],
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self):
        """Evaluate the model with detailed analysis"""
        print("Evaluating advanced model on test set...")
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Predictions
        predictions = self.model.predict(self.X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Analyze prediction distribution
        unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
        print(f"\nPrediction distribution: {dict(zip([self.class_names[i] for i in unique_pred], pred_counts))}")
        
        # Check prediction confidence
        max_confidences = np.max(predictions, axis=1)
        print(f"Average prediction confidence: {max_confidences.mean():.3f}")
        print(f"Min prediction confidence: {max_confidences.min():.3f}")
        print(f"Max prediction confidence: {max_confidences.max():.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))
        
        return test_accuracy, predictions, y_pred
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Advanced Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Advanced Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Advanced Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def save_model(self, model_path='advanced_cricket_shot_classifier.keras'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"Advanced model saved to {model_path}")
        else:
            print("No model to save")

def main():
    """Main function to run the advanced pipeline"""
    
    # Initialize classifier
    classifier = AdvancedCricketShotClassifier(
        data_dir='.',  # Current directory containing shot folders
        img_size=(224, 224),
        batch_size=32
    )
    
    # Load and preprocess data
    classifier.load_and_preprocess_data()
    
    # Create data generators
    classifier.create_data_generators()
    
    # Build advanced model
    print("\nBuilding advanced model...")
    classifier.build_advanced_model(model_type='custom')  # or 'resnet'
    
    # Train model
    print("\nTraining advanced model...")
    classifier.train_model(epochs=40, patience=20)
    
    # Evaluate model
    print("\nEvaluating advanced model...")
    test_accuracy, predictions, y_pred = classifier.evaluate_model()
    
    # Plot results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(y_pred)
    
    # Save the model
    classifier.save_model()
    
    print(f"\nFinal Advanced Test Accuracy: {test_accuracy:.4f}")
    print("Advanced training completed!")

if __name__ == "__main__":
    main() 
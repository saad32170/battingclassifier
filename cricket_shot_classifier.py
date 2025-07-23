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
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class CricketShotClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the Cricket Shot Classifier
        
        Args:
            data_dir: Directory containing shot type folders
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        self.num_classes = len(self.class_names)
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            print(f"Processing {class_name} images...")
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
                            
                            images.append(img)
                            labels.append(class_idx)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        
        print(f"Loaded {len(self.images)} images")
        print(f"Class distribution: {np.bincount(self.labels)}")
        
        return self.images, self.labels
    
    def create_data_generators(self, test_size=0.2, val_size=0.2):
        """Create train, validation, and test splits with data augmentation"""
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels, test_size=test_size, 
            stratify=self.labels, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, 
            stratify=y_temp, random_state=42
        )
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
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
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def build_model(self, model_type='custom'):
        """Build the classification model"""
        
        if model_type == 'resnet':
            # Transfer learning with ResNet50V2
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
            # Freeze base model
            base_model.trainable = False
            
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
        elif model_type == 'efficientnet':
            # Transfer learning with EfficientNet
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
            base_model.trainable = False
            
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
        else:
            # Custom CNN architecture
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model Summary:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=50, patience=10):
        """Train the model with early stopping and callbacks"""
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self):
        """Evaluate the model on test set"""
        print("Evaluating model on test set...")
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Predictions
        predictions = self.model.predict(self.X_test)
        y_pred = np.argmax(predictions, axis=1)
        
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
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
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
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def predict_single_image(self, image_path):
        """Predict shot type for a single image"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return self.class_names[predicted_class], confidence, prediction[0]
    
    def save_model(self, model_path='cricket_shot_classifier.h5'):
        """Save the trained model"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='cricket_shot_classifier.h5'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

def main():
    """Main function to run the complete pipeline"""
    
    # Initialize classifier
    classifier = CricketShotClassifier(
        data_dir='.',  # Current directory containing shot folders
        img_size=(224, 224),
        batch_size=32
    )
    
    # Load and preprocess data
    classifier.load_and_preprocess_data()
    
    # Create data generators
    classifier.create_data_generators()
    
    # Build model (try different architectures)
    print("\nBuilding model...")
    classifier.build_model(model_type='custom')  # or 'resnet', 'efficientnet'
    
    # Train model
    print("\nTraining model...")
    classifier.train_model(epochs=50, patience=10)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_accuracy, predictions, y_pred = classifier.evaluate_model()
    
    # Plot results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(y_pred)
    
    # Save model
    classifier.save_model()
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 
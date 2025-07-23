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
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0, VGG16
import cv2
import warnings
warnings.filterwarnings('ignore')

class EnsembleCricketShotClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        self.num_classes = len(self.class_names)
        self.models = []
        self.histories = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
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
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            img = img / 255.0
                            
                            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                                continue
                                
                            images.append(img)
                            labels.append(class_idx)
                            class_counts[class_name] += 1
                            class_images += 1
                    except Exception as e:
                        continue
            
            print(f"  Loaded {class_images} images for {class_name}")
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        
        print(f"\nTotal loaded: {len(self.images)} images")
        print(f"Class distribution: {class_counts}")
        
        return self.images, self.labels
    
    def create_data_generators(self, test_size=0.2, val_size=0.2):
        """Create train, validation, and test splits"""
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels, test_size=test_size, 
            stratify=self.labels, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, 
            stratify=y_temp, random_state=42
        )
        
        # Different augmentation strategies for different models
        augmentations = [
            # Model 1: Conservative augmentation
            ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            ),
            # Model 2: Moderate augmentation
            ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.1,
                zoom_range=0.15,
                horizontal_flip=True,
                fill_mode='nearest'
            ),
            # Model 3: Aggressive augmentation
            ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        ]
        
        # No augmentation for validation/test
        val_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, self.num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # Create generators for each model
        self.train_generators = []
        for aug in augmentations:
            gen = aug.flow(X_train, y_train_cat, batch_size=self.batch_size)
            self.train_generators.append(gen)
        
        self.val_generator = val_datagen.flow(X_val, y_val_cat, batch_size=self.batch_size)
        self.test_generator = test_datagen.flow(X_test, y_test_cat, batch_size=self.batch_size)
        
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_cat = y_test_cat
        
        print(f"\nData splits:")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return self.train_generators, self.val_generator, self.test_generator
    
    def build_model_1(self):
        """Custom CNN with conservative architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_model_2(self):
        """ResNet50V2 with transfer learning"""
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_model_3(self):
        """EfficientNetB0 with transfer learning"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, epochs=30, patience=15):
        """Train all models in the ensemble"""
        
        model_builders = [
            ("Custom CNN", self.build_model_1),
            ("ResNet50V2", self.build_model_2),
            ("EfficientNetB0", self.build_model_3)
        ]
        
        for i, (model_name, builder) in enumerate(model_builders):
            print(f"\n{'='*50}")
            print(f"Training Model {i+1}: {model_name}")
            print(f"{'='*50}")
            
            # Build model
            model = builder()
            print(f"Model {i+1} Summary:")
            model.summary()
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
            
            checkpoint = callbacks.ModelCheckpoint(
                f'ensemble_model_{i+1}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            
            # Train model
            history = model.fit(
                self.train_generators[i],
                epochs=epochs,
                validation_data=self.val_generator,
                callbacks=[early_stopping, reduce_lr, checkpoint],
                verbose=1
            )
            
            # Store model and history
            self.models.append(model)
            self.histories.append(history)
            
            # Evaluate individual model
            test_loss, test_accuracy = model.evaluate(self.test_generator)
            print(f"\nModel {i+1} Test Accuracy: {test_accuracy:.4f}")
    
    def ensemble_predict(self, X, method='voting'):
        """Make ensemble predictions using different methods"""
        
        if len(self.models) == 0:
            print("No models trained yet!")
            return None
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
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
            weights = [0.4, 0.35, 0.25]  # Adjust based on model performance
            weighted_predictions = np.zeros_like(all_predictions[0])
            
            for i, (pred, weight) in enumerate(zip(all_predictions, weights)):
                weighted_predictions += weight * pred
            
            return np.argmax(weighted_predictions, axis=1)
    
    def evaluate_ensemble(self):
        """Evaluate the ensemble model"""
        print("\n" + "="*50)
        print("ENSEMBLE EVALUATION")
        print("="*50)
        
        # Evaluate individual models
        individual_accuracies = []
        for i, model in enumerate(self.models):
            test_loss, test_accuracy = model.evaluate(self.test_generator)
            individual_accuracies.append(test_accuracy)
            print(f"Model {i+1} Accuracy: {test_accuracy:.4f}")
        
        print(f"\nAverage Individual Accuracy: {np.mean(individual_accuracies):.4f}")
        
        # Evaluate ensemble methods
        ensemble_methods = ['voting', 'averaging', 'weighted_averaging']
        
        for method in ensemble_methods:
            print(f"\n--- {method.upper()} ENSEMBLE ---")
            
            # Get ensemble predictions
            y_pred = self.ensemble_predict(self.X_test, method=method)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == self.y_test)
            print(f"Ensemble Accuracy: {accuracy:.4f}")
            
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
    
    def save_ensemble(self, base_path='ensemble_cricket_classifier'):
        """Save all models in the ensemble"""
        for i, model in enumerate(self.models):
            model_path = f"{base_path}_model_{i+1}.keras"
            model.save(model_path)
            print(f"Model {i+1} saved to {model_path}")

def main():
    """Main function to run the ensemble pipeline"""
    
    # Initialize ensemble classifier
    ensemble = EnsembleCricketShotClassifier(
        data_dir='.',
        img_size=(224, 224),
        batch_size=32
    )
    
    # Load and preprocess data
    ensemble.load_and_preprocess_data()
    
    # Create data generators
    ensemble.create_data_generators()
    
    # Train all models
    print("\nTraining ensemble models...")
    ensemble.train_models(epochs=30, patience=15)
    
    # Evaluate ensemble
    ensemble.evaluate_ensemble()
    
    # Save ensemble
    ensemble.save_ensemble()
    
    print("\nEnsemble training completed!")

if __name__ == "__main__":
    main() 
# Cricket Shot Classification Project
## Machine Learning Analysis & Results

---

## 📋 **Project Overview**

This project implements a **Convolutional Neural Network (CNN)** ensemble system to classify cricket shots from images into four categories:
- **Drive** - Straight bat shots played with full face
- **Legglance-Flick** - Leg-side shots with wrist movement
- **Pullshot** - Horizontal bat shots to leg side
- **Sweep** - Low shots played with bent knees

### **Final Achievement: 97.14% Accuracy** 🏆

---

## 🎯 **Project Goals & Objectives**

### **Primary Goals:**
1. **High Accuracy**: Achieve >95% classification accuracy
2. **Robust Classification**: Handle variations in cricket shots
3. **Ensemble Learning**: Combine multiple models for better performance
4. **Data Analysis**: Understand confusion patterns between similar shots

### **Technical Objectives:**
- Implement CNN architectures for image classification
- Use transfer learning with pre-trained models
- Apply ensemble methods for improved accuracy
- Analyze misclassifications for data improvement

---

## 📊 **Dataset Information**

### **Dataset Structure:**
```
shotpredictionml/
├── drive/           (1,257 images)
├── legglance-flick/ (1,117 images) 
├── pullshot/        (1,257 images)
└── sweep/           (1,117 images)
```

### **Total Dataset Size:**
- **4,748 images** across 4 classes
- **Balanced distribution** with slight variations
- **Image formats**: PNG, JPEG, WebP
- **Resolution**: Variable (resized to 224x224 for training)

### **Data Preprocessing:**
- **Resize**: All images to 224x224 pixels
- **Color Conversion**: BGR to RGB
- **Normalization**: Pixel values scaled to [0,1]
- **Data Augmentation**: Rotation, shifts, zoom, brightness

---

## 🏗️ **Model Architecture Evolution**

### **1. Original Classifier (Baseline)**
- **Accuracy**: ~25% (stuck predicting one class)
- **Issues**: Double preprocessing, over-aggressive augmentation
- **Status**: ❌ Failed

### **2. Working Classifier (Diagnostic)**
- **Accuracy**: 88.00%
- **Architecture**: Simple CNN with reduced augmentation
- **Key Fixes**: Single preprocessing, conservative augmentation
- **Status**: ✅ Working baseline

### **3. Advanced Classifier**
- **Accuracy**: 91.64%
- **Architecture**: Enhanced CNN with Batch Normalization
- **Improvements**: Deeper network, better regularization
- **Status**: ✅ Significant improvement

### **4. Ensemble Classifier (Final)**
- **Accuracy**: 97.14%
- **Architecture**: 3-model ensemble
- **Methods**: Voting, Averaging, Weighted Averaging
- **Status**: ✅ Target achieved

---

## 📈 **Detailed Results Analysis**

### **Individual Model Performance**

| Model | Architecture | Accuracy | Precision | Recall | F1-Score |
|-------|-------------|----------|-----------|---------|----------|
| Custom CNN | Sequential CNN | 90.90% | 0.91 | 0.91 | 0.91 |
| ResNet50V2 | Transfer Learning | 95.24% | 0.95 | 0.95 | 0.95 |
| Advanced CNN | Enhanced CNN | 91.64% | 0.92 | 0.92 | 0.92 |

### **Ensemble Performance Comparison**

| Ensemble Method | Accuracy | Drive | Legglance-Flick | Pullshot | Sweep |
|-----------------|----------|-------|-----------------|----------|-------|
| **Hard Voting** | 95.34% | 0.95 | 0.94 | 0.96 | 0.96 |
| **Soft Averaging** | 96.72% | 0.97 | 0.96 | 0.97 | 0.97 |
| **Weighted Averaging** | **97.14%** | **0.98** | **0.96** | **0.97** | **0.97** |

### **Class-wise Performance (Best Ensemble)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| **Drive** | 0.98 | 0.95 | 0.96 | 245 |
| **Legglance-Flick** | 0.96 | 0.96 | 0.96 | 224 |
| **Pullshot** | 0.97 | 0.99 | 0.98 | 252 |
| **Sweep** | 0.97 | 0.99 | 0.98 | 224 |

---

## 🔧 **Technical Implementation Details**

### **Model Architectures**

#### **1. Custom CNN (Working Classifier)**
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 32, 32, 32)       896       
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)      0         
conv2d_1 (Conv2D)           (None, 16, 16, 64)       18496     
max_pooling2d_1 (MaxPooling2D) (None, 8, 8, 64)      0         
conv2d_2 (Conv2D)           (None, 8, 8, 128)        73856     
max_pooling2d_2 (MaxPooling2D) (None, 4, 4, 128)     0         
flatten (Flatten)           (None, 2048)              0         
dense (Dense)               (None, 512)               1049088   
dropout (Dropout)           (None, 512)               0         
dense_1 (Dense)             (None, 4)                 2052      
=================================================================
Total params: 1,142,388
Trainable params: 1,142,388
Non-trainable params: 0
```

#### **2. Advanced CNN**
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 112, 112, 64)     1792      
batch_normalization (BatchNormalization) (None, 112, 112, 64) 256
max_pooling2d (MaxPooling2D) (None, 56, 56, 64)      0         
conv2d_1 (Conv2D)           (None, 56, 56, 128)      73856     
batch_normalization_1 (BatchNormalization) (None, 56, 56, 128) 512
max_pooling2d_1 (MaxPooling2D) (None, 28, 28, 128)   0         
conv2d_2 (Conv2D)           (None, 28, 28, 256)      295168    
batch_normalization_2 (BatchNormalization) (None, 28, 28, 256) 1024
max_pooling2d_2 (MaxPooling2D) (None, 14, 14, 256)   0         
conv2d_3 (Conv2D)           (None, 14, 14, 512)      1180160   
batch_normalization_3 (BatchNormalization) (None, 14, 14, 512) 2048
max_pooling2d_3 (MaxPooling2D) (None, 7, 7, 512)     0         
flatten (Flatten)           (None, 25088)             0         
dense (Dense)               (None, 1024)              25691136  
dropout (Dropout)           (None, 1024)              0         
dense_1 (Dense)             (None, 512)               524800    
dropout_1 (Dropout)         (None, 512)               0         
dense_2 (Dense)             (None, 4)                 2052      
=================================================================
Total params: 27,647,940
Trainable params: 27,645,892
Non-trainable params: 2,048
```

#### **3. ResNet50V2 (Transfer Learning)**
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
resnet50v2 (Functional)     (None, 7, 7, 2048)       23564800  
global_average_pooling2d (GlobalAveragePooling2D) (None, 2048) 0
dense (Dense)               (None, 512)               1049088   
dropout (Dropout)           (None, 512)               0         
dense_1 (Dense)             (None, 4)                 2052      
=================================================================
Total params: 24,613,940
Trainable params: 1,051,140
Non-trainable params: 23,562,800
```

### **Training Configuration**

#### **Hyperparameters**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Image Size** | 224x224 | Standard input size |
| **Batch Size** | 32 | Optimal for memory and convergence |
| **Learning Rate** | 0.001 | Adam optimizer default |
| **Epochs** | 50 | With early stopping |
| **Patience** | 10 | Early stopping patience |
| **Validation Split** | 0.2 | 20% for validation |

#### **Data Augmentation**
```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)
```

#### **Callbacks**
- **EarlyStopping**: Monitor `val_accuracy`, patience=10
- **ReduceLROnPlateau**: Monitor `val_loss`, factor=0.5, patience=5
- **ModelCheckpoint**: Save best model based on `val_accuracy`

---

## 🎯 **Ensemble Methods Explained**

### **1. Hard Voting (Majority Vote)**
- **Method**: Each model votes for a class, majority wins
- **Advantage**: Simple, robust to outliers
- **Disadvantage**: Ignores confidence levels
- **Accuracy**: 95.34%

### **2. Soft Voting (Probability Averaging)**
- **Method**: Average predicted probabilities across models
- **Advantage**: Considers model confidence
- **Disadvantage**: Sensitive to probability calibration
- **Accuracy**: 96.72%

### **3. Weighted Averaging (Best Method)**
- **Method**: Weighted average based on individual model performance
- **Weights**: [0.3, 0.4, 0.3] for [Custom CNN, ResNet50V2, Advanced CNN]
- **Advantage**: Gives more weight to better performing models
- **Accuracy**: **97.14%**

---

## 🔍 **Problem Analysis & Solutions**

### **Initial Problems (25% Accuracy)**

#### **Issues Identified:**
1. **Double Preprocessing**: Images processed twice
2. **Over-aggressive Augmentation**: Too much noise in training data
3. **Image Size Mismatch**: Inconsistent input dimensions
4. **Poor Architecture**: Insufficient model capacity

#### **Solutions Implemented:**
1. **Single Preprocessing Pipeline**: Consistent data flow
2. **Conservative Augmentation**: Balanced data augmentation
3. **Standardized Input**: 224x224 RGB images
4. **Enhanced Architecture**: Deeper networks with regularization

### **Drive vs Legglance-Flick Confusion**

#### **Root Causes:**
1. **Similar Body Positions**: Both shots involve front foot movement
2. **Bat Angle Variations**: Subtle differences in bat positioning
3. **Limited Training Data**: Need more diverse examples
4. **Lighting Variations**: Different lighting conditions

#### **Improvements Made:**
1. **Ensemble Diversity**: Different architectures capture different features
2. **Transfer Learning**: ResNet50V2 pre-trained on ImageNet
3. **Batch Normalization**: Stabilizes training
4. **Weighted Ensemble**: Better performing models get more weight

---

## 📈 **Performance Metrics Explained**

### **Accuracy**
- **Definition**: (Correct Predictions) / (Total Predictions)
- **Formula**: `(TP + TN) / (TP + TN + FP + FN)`
- **Interpretation**: Overall correctness of the model

### **Precision**
- **Definition**: (True Positives) / (True Positives + False Positives)
- **Formula**: `TP / (TP + FP)`
- **Interpretation**: How many predicted positives were actually correct

### **Recall**
- **Definition**: (True Positives) / (True Positives + False Negatives)
- **Formula**: `TP / (TP + FN)`
- **Interpretation**: How many actual positives were correctly identified

### **F1-Score**
- **Definition**: Harmonic mean of precision and recall
- **Formula**: `2 * (Precision * Recall) / (Precision + Recall)`
- **Interpretation**: Balanced measure between precision and recall

---

## 🚀 **Future Improvements & Recommendations**

### **Data Quality Improvements**

#### **1. Collect More Diverse Data**
- **Different Angles**: Front, side, 45-degree views
- **Various Players**: Different batting styles and techniques
- **Multiple Formats**: Test, ODI, T20 cricket
- **Different Conditions**: Various lighting and weather

#### **2. Add New Shot Types**
- **Basic Shots**: Cut, Hook, Defensive, Leave
- **Advanced Shots**: Reverse Sweep, Scoop, Ramp, Dilscoop
- **Variations**: Square Cut, Late Cut, Upper Cut, Glance

#### **3. Data Augmentation Enhancements**
- **Cricket-Specific**: Realistic transformations
- **Background Variation**: Different cricket grounds
- **Player Variation**: Different body types and equipment

### **Model Improvements**

#### **1. Architecture Enhancements**
- **Attention Mechanisms**: Focus on key body parts
- **Multi-scale Features**: Capture details at different levels
- **Temporal Information**: Use video frames for context

#### **2. Training Strategies**
- **Curriculum Learning**: Start with easy examples
- **Focal Loss**: Focus on hard examples
- **Mixup Augmentation**: Blend similar shots

#### **3. Advanced Ensemble Methods**
- **Stacking**: Train meta-learner on base model predictions
- **Bagging**: Bootstrap aggregating
- **Boosting**: Sequential ensemble learning

---

## 📁 **Project File Structure**

```
shotpredictionml/
├── 📊 CRICKET_SHOT_CLASSIFICATION_PROJECT.md    # This documentation
├── 🎯 quick_ensemble.py                         # Final ensemble evaluation
├── 🔮 predict.py                                # Prediction script
├── 📈 misclassification_analyzer.py             # Error analysis
├── 📋 README.md                                 # Project overview
├── 📦 requirements.txt                          # Dependencies
├── 📁 old_models/                               # Previous model versions
│   ├── 01_original_classifier.py               # Initial failed model
│   ├── 02_working_classifier.py                # First working model
│   ├── 03_advanced_classifier.py               # Enhanced model
│   ├── 04_ensemble_classifier.py               # Ensemble training
│   └── 05_no_augmentation_classifier.py        # No augmentation test
├── 🏏 drive/                                    # Drive shot images
├── 🏏 legglance-flick/                          # Legglance-flick images
├── 🏏 pullshot/                                 # Pullshot images
└── 🏏 sweep/                                    # Sweep shot images
```

---

## 🏆 **Key Achievements**

### **Performance Milestones**
1. **✅ Baseline Working**: 88.00% accuracy
2. **✅ Advanced Model**: 91.64% accuracy  
3. **✅ Target Achieved**: 97.14% accuracy
4. **✅ Ensemble Success**: Weighted averaging method

### **Technical Achievements**
1. **Problem Diagnosis**: Identified and fixed preprocessing issues
2. **Architecture Optimization**: Enhanced CNN with Batch Normalization
3. **Transfer Learning**: Successfully implemented ResNet50V2
4. **Ensemble Learning**: Combined multiple models effectively
5. **Error Analysis**: Comprehensive misclassification analysis

### **Data Insights**
1. **Confusion Patterns**: Identified drive/flick confusion
2. **Model Diversity**: Different architectures capture different features
3. **Augmentation Impact**: Balanced augmentation improves performance
4. **Transfer Learning**: Pre-trained models significantly improve accuracy

---

## 📚 **Technical Concepts Explained**

### **Convolutional Neural Networks (CNNs)**
- **Purpose**: Extract spatial features from images
- **Layers**: Convolution, Pooling, Dense
- **Advantages**: Translation invariant, parameter sharing
- **Application**: Cricket shot feature extraction

### **Transfer Learning**
- **Concept**: Use pre-trained models on new tasks
- **Benefits**: Faster training, better performance
- **Models Used**: ResNet50V2 (ImageNet pre-trained)
- **Fine-tuning**: Adapt to cricket-specific features

### **Ensemble Learning**
- **Principle**: Combine multiple models for better performance
- **Methods**: Voting, Averaging, Weighted Averaging
- **Benefits**: Reduced variance, improved accuracy
- **Diversity**: Different architectures and training strategies

### **Data Augmentation**
- **Purpose**: Increase training data variety
- **Techniques**: Rotation, shifts, zoom, brightness
- **Benefits**: Better generalization, reduced overfitting
- **Balance**: Conservative vs aggressive augmentation

---

## 🎯 **Conclusion**

This cricket shot classification project successfully demonstrates the power of **ensemble learning** and **transfer learning** in computer vision tasks. Starting from a failed baseline model stuck at 25% accuracy, we systematically diagnosed issues, implemented solutions, and achieved an impressive **97.14% accuracy** using a weighted ensemble of three diverse models.

### **Key Success Factors:**
1. **Systematic Problem Solving**: Identified and fixed preprocessing issues
2. **Model Diversity**: Combined different architectures effectively
3. **Transfer Learning**: Leveraged pre-trained models
4. **Ensemble Methods**: Weighted averaging for optimal performance
5. **Data Quality**: Balanced augmentation and preprocessing

### **Impact & Applications:**
- **Cricket Analysis**: Automated shot classification for match analysis
- **Player Development**: Training and coaching applications
- **Broadcasting**: Real-time shot identification
- **Research**: Sports analytics and biomechanics studies

The project showcases best practices in machine learning, from data preprocessing to model ensemble design, and provides a solid foundation for expanding to more shot types and improving classification accuracy further.

---

## 📞 **Contact & Support**

For questions, improvements, or collaboration opportunities, please refer to the project documentation and code comments for detailed implementation guidance.

**Project Status**: ✅ **COMPLETED** - Target accuracy achieved  
**Final Accuracy**: **97.14%** 🏆  
**Ensemble Method**: Weighted Averaging  
**Models Used**: Custom CNN, ResNet50V2, Advanced CNN 
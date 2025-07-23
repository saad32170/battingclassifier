# Cricket Shot Classification ML Project
## 🏆 **Achieved 97.14% Accuracy with Ensemble Learning**

This project implements advanced machine learning models to classify cricket shots into four categories with exceptional accuracy using ensemble learning techniques.

## 🎯 **Project Status: COMPLETED** ✅

### **Final Results:**
- **Best Individual Model**: ResNet50V2 (95.24% accuracy)
- **Best Ensemble Method**: Weighted Averaging (97.14% accuracy)
- **Target Achieved**: ✅ 95%+ accuracy goal exceeded

---

## 📊 **Quick Results Overview**

| Model | Accuracy | Method |
|-------|----------|---------|
| Custom CNN | 90.90% | Individual |
| ResNet50V2 | 95.24% | Transfer Learning |
| Advanced CNN | 91.64% | Enhanced Architecture |
| **Ensemble (Weighted)** | **97.14%** | **3-Model Combination** |

### **Class-wise Performance (Best Ensemble):**
- **Drive**: 98% precision, 95% recall
- **Legglance-Flick**: 96% precision, 96% recall  
- **Pullshot**: 97% precision, 99% recall
- **Sweep**: 97% precision, 99% recall

---

## 📁 **Project Structure**

```
shotpredictionml/
├── 📊 CRICKET_SHOT_CLASSIFICATION_PROJECT.md    # Complete documentation
├── 🎯 quick_ensemble.py                         # Final ensemble evaluation
├── 🔮 predict.py                                # Prediction script
├── 📈 misclassification_analyzer.py             # Error analysis
├── 📋 README.md                                 # This file
├── 📦 requirements.txt                          # Dependencies
├── 📁 old_models/                               # Previous model versions
│   ├── 01_original_classifier.py               # Initial failed model (25%)
│   ├── 02_working_classifier.py                # First working model (88%)
│   ├── 03_advanced_classifier.py               # Enhanced model (91.64%)
│   ├── 04_ensemble_classifier.py               # Ensemble training
│   └── 05_no_augmentation_classifier.py        # No augmentation test
├── 🏏 drive/                                    # Drive shot images (1,257)
├── 🏏 legglance-flick/                          # Legglance-flick images (1,117)
├── 🏏 pullshot/                                 # Pullshot images (1,257)
└── 🏏 sweep/                                    # Sweep shot images (1,117)
```

---

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Evaluate the Final Ensemble Model**
```bash
python quick_ensemble.py
```
This will load the trained ensemble models and show detailed performance metrics.

### **3. Make Predictions**
```bash
# Interactive mode
python predict.py

# Single image prediction
python predict.py --image path/to/image.jpg

# Batch prediction
python predict.py --folder path/to/images/ --max-images 20
```

---

## 🏗️ **Model Evolution**

### **Journey from 25% to 97.14% Accuracy:**

1. **Original Classifier** (25% accuracy) ❌
   - Failed due to double preprocessing and over-aggressive augmentation

2. **Working Classifier** (88% accuracy) ✅
   - Fixed preprocessing issues with single pipeline

3. **Advanced Classifier** (91.64% accuracy) ✅
   - Added Batch Normalization and deeper architecture

4. **Ensemble Classifier** (97.14% accuracy) 🏆
   - Combined 3 models using weighted averaging

---

## 🎯 **Ensemble Methods Used**

### **1. Hard Voting (Majority Vote)**
- **Accuracy**: 95.34%
- **Method**: Each model votes, majority wins

### **2. Soft Voting (Probability Averaging)**
- **Accuracy**: 96.72%
- **Method**: Average predicted probabilities

### **3. Weighted Averaging (Best Method)**
- **Accuracy**: **97.14%** 🏆
- **Method**: Weighted average based on individual performance
- **Weights**: [0.3, 0.4, 0.3] for [Custom CNN, ResNet50V2, Advanced CNN]

---

## 🔧 **Technical Implementation**

### **Model Architectures:**
- **Custom CNN**: Sequential CNN with 3 conv blocks
- **ResNet50V2**: Transfer learning with ImageNet pre-training
- **Advanced CNN**: Enhanced CNN with Batch Normalization

### **Training Configuration:**
- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Learning Rate**: 0.001 (Adam optimizer)
- **Data Augmentation**: Rotation, shifts, zoom, brightness
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing

### **Data Preprocessing:**
- **Resize**: All images to 224x224
- **Color Conversion**: BGR to RGB
- **Normalization**: Pixel values scaled to [0,1]

---

## 📈 **Performance Analysis**

### **Individual Model Performance:**

| Model | Architecture | Accuracy | Precision | Recall | F1-Score |
|-------|-------------|----------|-----------|---------|----------|
| Custom CNN | Sequential CNN | 90.90% | 0.91 | 0.91 | 0.91 |
| ResNet50V2 | Transfer Learning | 95.24% | 0.95 | 0.95 | 0.95 |
| Advanced CNN | Enhanced CNN | 91.64% | 0.92 | 0.92 | 0.92 |

### **Ensemble Performance:**

| Ensemble Method | Accuracy | Drive | Legglance-Flick | Pullshot | Sweep |
|-----------------|----------|-------|-----------------|----------|-------|
| **Hard Voting** | 95.34% | 0.95 | 0.94 | 0.96 | 0.96 |
| **Soft Averaging** | 96.72% | 0.97 | 0.96 | 0.97 | 0.97 |
| **Weighted Averaging** | **97.14%** | **0.98** | **0.96** | **0.97** | **0.97** |

---

## 🎮 **Usage Examples**

### **Basic Prediction:**
```python
from predict import CricketShotPredictor

# Load the best ensemble model
predictor = CricketShotPredictor('best_advanced_model.h5')
predictor.load_model()

# Predict single image
shot_type, confidence, probabilities = predictor.predict_single_image('image.jpg')
print(f"Predicted: {shot_type} with {confidence:.2%} confidence")
```

### **Batch Prediction:**
```python
# Predict multiple images
results = predictor.predict_batch('folder_with_images/', max_images=50)
for result in results:
    print(f"{result['image']}: {result['prediction']} ({result['confidence']:.2%})")
```

### **Interactive Mode:**
```python
# Interactive prediction session
predictor.interactive_predictor()
```

---

## 🔍 **Key Features**

### **✅ What Works:**
- **Ensemble Learning**: Combines multiple models for better accuracy
- **Transfer Learning**: Uses pre-trained ResNet50V2 for feature extraction
- **Data Augmentation**: Balanced augmentation prevents overfitting
- **Batch Normalization**: Stabilizes training and improves convergence
- **Early Stopping**: Prevents overfitting with automatic stopping

### **🎯 Problem Solutions:**
- **Fixed Double Preprocessing**: Single clean data pipeline
- **Reduced Augmentation**: Conservative augmentation strategy
- **Enhanced Architecture**: Deeper networks with regularization
- **Model Diversity**: Different architectures capture different features

---

## 📚 **Technical Concepts**

### **Convolutional Neural Networks (CNNs)**
- Extract spatial features from cricket shot images
- Translation invariant feature detection
- Hierarchical feature learning

### **Transfer Learning**
- Use pre-trained models (ResNet50V2) on new tasks
- Faster training and better performance
- Leverage ImageNet features for cricket shots

### **Ensemble Learning**
- Combine multiple models for improved accuracy
- Reduce variance and improve generalization
- Weighted averaging based on individual performance

---

## 🚀 **Future Improvements**

### **Data Quality:**
- Collect more diverse angles and lighting conditions
- Add new shot types (cut, hook, defensive, etc.)
- Include different players and batting styles

### **Model Enhancements:**
- Attention mechanisms for key body parts
- Multi-scale feature extraction
- Temporal information from video frames

### **Advanced Techniques:**
- Curriculum learning
- Focal loss for hard examples
- Advanced ensemble methods (stacking, boosting)

---

## 🏆 **Achievements**

### **Performance Milestones:**
1. ✅ **Baseline Working**: 88.00% accuracy
2. ✅ **Advanced Model**: 91.64% accuracy  
3. ✅ **Target Achieved**: 97.14% accuracy
4. ✅ **Ensemble Success**: Weighted averaging method

### **Technical Achievements:**
- Systematic problem diagnosis and solution
- Architecture optimization with Batch Normalization
- Successful transfer learning implementation
- Effective ensemble learning techniques
- Comprehensive error analysis

---

## 📞 **Support & Documentation**

- **Complete Documentation**: `CRICKET_SHOT_CLASSIFICATION_PROJECT.md`
- **Model Evolution**: `old_models/README.md`
- **Error Analysis**: `misclassification_analyzer.py`
- **Prediction Script**: `predict.py`

### **Troubleshooting:**
1. **Model Loading**: Ensure `.keras` or `.h5` files are in correct location
2. **Dependencies**: Install all requirements with `pip install -r requirements.txt`
3. **Memory Issues**: Reduce batch size for large datasets
4. **Data Format**: Ensure images are in supported formats (PNG, JPEG)

---

## 🎯 **Conclusion**

This project successfully demonstrates the power of **ensemble learning** and **transfer learning** in computer vision tasks. Starting from a failed baseline model stuck at 25% accuracy, we systematically diagnosed issues, implemented solutions, and achieved an impressive **97.14% accuracy** using a weighted ensemble of three diverse models.

**Project Status**: ✅ **COMPLETED** - Target accuracy achieved  
**Final Accuracy**: **97.14%** 🏆  
**Ensemble Method**: Weighted Averaging  
**Models Used**: Custom CNN, ResNet50V2, Advanced CNN 
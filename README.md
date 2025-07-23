# Cricket Shot Classification ML Project

This project implements machine learning models to classify cricket shots into four categories: drive, legglance-flick, pullshot, and sweep.

## Project Structure

```
shotpredictionml/
├── drive/                 # Drive shot images
├── legglance-flick/       # Leg glance/flick shot images  
├── pullshot/             # Pull shot images
├── sweep/                # Sweep shot images
├── cricket_shot_classifier.py      # Original classifier (80% accuracy)
├── improved_classifier.py          # Improved version (fixed)
├── quick_fix_classifier.py         # Quick fix version
├── no_augmentation_classifier.py   # For pre-augmented data
├── predict.py                      # Prediction script
├── test_setup.py                   # Setup testing
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model

**Option A: Original Working Classifier (80% accuracy)**
```bash
python cricket_shot_classifier.py
```

**Option B: Quick Fix Classifier**
```bash
python quick_fix_classifier.py
```

**Option C: No Augmentation Classifier (if images are pre-augmented)**
```bash
python no_augmentation_classifier.py
```

### 3. Use the Trained Model for Predictions

**Interactive Mode:**
```bash
python predict.py
```

**Single Image Prediction:**
```bash
python predict.py --image path/to/image.jpg
```

**Batch Prediction:**
```bash
python predict.py --folder path/to/images/ --max-images 20
```

**Specify Model:**
```bash
python predict.py --model my_model.h5 --image path/to/image.jpg
```

## Model Training Options

### 1. Original Classifier (`cricket_shot_classifier.py`)
- **Accuracy**: ~80%
- **Architecture**: Custom CNN
- **Augmentation**: Moderate
- **Best for**: Getting started, baseline performance

### 2. Quick Fix Classifier (`quick_fix_classifier.py`)
- **Accuracy**: ~80%
- **Architecture**: Custom CNN (fixed version)
- **Augmentation**: Moderate
- **Best for**: Reliable training with good accuracy

### 3. No Augmentation Classifier (`no_augmentation_classifier.py`)
- **Accuracy**: Varies (optimized for pre-augmented data)
- **Architecture**: EfficientNetB0
- **Augmentation**: None (data is pre-augmented)
- **Best for**: Datasets with already augmented images

## What Happens After Training?

Once you have a trained model (`.h5` file), you can:

### 1. Make Predictions
```python
from predict import CricketShotPredictor

# Load model
predictor = CricketShotPredictor('my_model.h5')
predictor.load_model()

# Predict single image
shot_type, confidence, probabilities = predictor.predict_single_image('image.jpg')
```

### 2. Batch Predictions
```python
# Predict multiple images
results = predictor.predict_batch('folder_with_images/', max_images=50)
```

### 3. Interactive Mode
```python
# Interactive prediction session
predictor.interactive_predictor()
```

### 4. Deploy the Model
You can integrate the trained model into:
- Web applications (Flask, Django)
- Mobile apps
- Real-time video analysis
- API services

## Model Files

After training, you'll get:
- `*.h5` - The trained model file
- Training plots (accuracy/loss curves)
- Confusion matrix
- Classification report

## Troubleshooting

### Low Accuracy Issues
1. **Check data preprocessing**: Ensure images are properly loaded and normalized
2. **Try different architectures**: Switch between custom CNN, ResNet, or EfficientNet
3. **Adjust augmentation**: If images are pre-augmented, use `no_augmentation_classifier.py`
4. **Check class balance**: Ensure all classes have similar numbers of images

### Common Errors
- **Model not found**: Make sure to train a model first
- **Memory issues**: Reduce batch size or image size
- **Import errors**: Install all requirements with `pip install -r requirements.txt`

## Performance Tips

1. **Use GPU**: Training is much faster with GPU support
2. **Batch size**: Adjust based on your memory (16-64 typically works)
3. **Image size**: 224x224 is standard, larger sizes need more memory
4. **Early stopping**: Models typically converge in 20-50 epochs

## Next Steps

After achieving good accuracy:
1. **Fine-tune**: Try different architectures or hyperparameters
2. **Deploy**: Create a web interface or API
3. **Real-time**: Integrate with video streams
4. **Expand**: Add more shot types or improve accuracy

## Example Usage

```python
# Complete workflow example
from predict import CricketShotPredictor

# 1. Train model (run one of the training scripts first)
# python quick_fix_classifier.py

# 2. Load and use model
predictor = CricketShotPredictor('quick_fix_classifier.h5')
predictor.load_model()

# 3. Make predictions
shot_type, confidence, probs = predictor.predict_single_image('test_image.jpg')
print(f"Predicted: {shot_type} with {confidence:.2%} confidence")
```

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your data structure matches the expected format
3. Ensure all dependencies are installed
4. Try the test setup script: `python test_setup.py` 
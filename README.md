# 🐱🐶 Pet Classifier

A pet classifier that distinguishes between cats and dogs using artificial intelligence. Basically, you give it a photo and it tells you if it's a cat or a dog.

## What does it do?

This project is a web API that:
- Receives pet images
- Uses a trained deep learning model to classify whether the image is of a cat or a dog
- Returns the prediction with confidence level

## How was it built?

### The Model
- **Architecture**: Convolutional Neural Network (CNN) with PyTorch
- **Training**: 15 epochs with cats and dogs dataset
- **Features**:
  - 4 convolutional layers
  - Dropout to prevent overfitting
  - Image normalization
  - Data augmentation

### The API
- **Backend**: Django + Django REST Framework
- **Endpoint**: `POST /api/predict/`
- **Input**: Image (multipart/form-data format)
- **Output**: JSON with prediction and probabilities

## 🚀 How to run it locally

### Prerequisites
- Python 3.8+
- pip

### 1. Clone and setup
```bash
git clone
cd petClassifier
```

### 2. Create virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (optional)
If you want to train your own model:
```bash
python train.py
```
**Note**: You need a dataset with this structure:
```
dataset/
├── train/
│   ├── cat/
│   └── dog/
└── validation/
    ├── cat/
    └── dog/
```

### 5. Run the server
```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/api/predict/`

## 📡 How to use the API

### With curl:
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:8000/api/predict/
```

### With Python:
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8000/api/predict/', files=files)
    print(response.json())
```

### Example response:
```json
{
    "predicted_class": "dog",
    "confidence": "0.9234",
    "probabilities": {
        "cat": "0.0766",
        "dog": "0.9234"
    }
}
```

## 📁 Project structure
```
petClassifier/
├── train.py              # Training script
├── pet_classifier.pth    # Trained model
├── requirements.txt      # Dependencies
├── manage.py            # Django management
├── pet_classifier_backend/  # Django configuration
└── prediction_api/      # Prediction app
    ├── views.py         # Prediction endpoint
    ├── model_loader.py  # Model loading
    └── urls.py          # API URLs
```

# 🐱🐶 Pet Classifier

Un clasificador de mascotas que distingue entre gatos y perros usando inteligencia artificial. Básicamente, le pasas una foto y te dice si es un gato o un perro.

## ¿Qué hace?

Este proyecto es una API web que:
- Recibe imágenes de mascotas
- Usa un modelo de deep learning entrenado para clasificar si la imagen es de un gato o un perro
- Te devuelve la predicción con el nivel de confianza

## ¿Cómo se hizo?

### El Modelo
- **Arquitectura**: Red Neuronal Convolucional (CNN) con PyTorch
- **Entrenamiento**: 15 épocas con dataset de gatos y perros
- **Características**:
  - 4 capas convolucionales
  - Dropout para evitar overfitting
  - Normalización de imágenes
  - Data augmentation

### La API
- **Backend**: Django + Django REST Framework
- **Endpoint**: `POST /api/predict/`
- **Entrada**: Imagen (formato multipart/form-data)
- **Salida**: JSON con predicción y probabilidades

## 🚀 Cómo correrlo localmente

### Prerrequisitos
- Python 3.8+
- pip

### 1. Clonar y configurar
```bash
git clone
cd petClassifier
```

### 2. Crear entorno virtual
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Entrenar el modelo (opcional)
Si quieres entrenar tu propio modelo:
```bash
python train.py
```
**Nota**: Necesitas un dataset con estructura:
```
dataset/
├── train/
│   ├── cat/
│   └── dog/
└── validation/
    ├── cat/
    └── dog/
```

### 5. Ejecutar el servidor
```bash
python manage.py runserver
```

La API estará disponible en `http://localhost:8000/api/predict/`

## 📡 Cómo usar la API

### Con curl:
```bash
curl -X POST -F "image=@tu_imagen.jpg" http://localhost:8000/api/predict/
```

### Con Python:
```python
import requests

with open('imagen.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8000/api/predict/', files=files)
    print(response.json())
```

### Respuesta ejemplo:
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

## 📁 Estructura del proyecto
```
petClassifier/
├── train.py              # Script de entrenamiento
├── pet_classifier.pth    # Modelo entrenado
├── requirements.txt      # Dependencias
├── manage.py            # Django management
├── pet_classifier_backend/  # Configuración Django
└── prediction_api/      # App de predicciones
    ├── views.py         # Endpoint de predicción
    ├── model_loader.py  # Carga del modelo
    └── urls.py          # URLs de la API
```

# prediction_api/model_loader.py

import torch
import torch.nn as nn
from torchvision import transforms
import os

# --- Configuración del Modelo (DEBE COINCIDIR CON train.py) ---
IMAGE_SIZE = (150, 150)
MODEL_NAME = 'pet_classifier.pth' # Nombre de tu archivo .pth

# Determina el directorio base del proyecto (donde está manage.py y pet_classifier.pth)
# Es un nivel por encima del directorio donde se encuentra este archivo (prediction_api)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# --- Definición del Modelo (Red Neuronal Convolucional - CNN) ---
# Esta clase DEBE ser idéntica a la que usaste en train.py para cargar el estado.
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Ajusta el tamaño de entrada de la capa lineal según el IMAGE_SIZE y pools
        self.fc1 = nn.Linear(256 * (IMAGE_SIZE[0] // 16) * (IMAGE_SIZE[1] // 16), 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1) 
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# --- Cargar el Modelo ---
# Se utiliza una variable para el dispositivo.
# Es crucial que la carga del modelo se haga aquí y no dentro de la clase para evitar recargas.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Variable para almacenar el modelo cargado
_loaded_model = None

def get_model():
    """
    Función para obtener el modelo. Lo carga solo la primera vez.
    """
    global _loaded_model
    if _loaded_model is None:
        model = CatDogClassifier().to(DEVICE)
        try:
            # Cargar el state_dict (solo los pesos)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval() # Poner el modelo en modo evaluación
            _loaded_model = model
            print(f"Modelo cargado exitosamente desde {MODEL_PATH} en {DEVICE}.")
        except FileNotFoundError:
            print(f"Error: El archivo del modelo '{MODEL_NAME}' no se encontró en '{MODEL_PATH}'.")
            print("Asegúrate de que 'pet_classifier.pth' esté en el mismo directorio que 'manage.py'.")
            _loaded_model = None # Asegurarse de que sea None si falla
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            _loaded_model = None # Asegurarse de que sea None si falla
    return _loaded_model

# --- Transformaciones para la Predicción ---
# Deben ser las mismas que las de validación/prueba en train.py
preprocess_image = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mapeo de índices a nombres de clases (asumiendo 0 para gato, 1 para perro)
# Revisa tu 'print("Índice de clases:", train_dataset.class_to_idx)' en train.py
# para asegurarte de que este mapeo es correcto.
CLASS_NAMES = {0: 'cat', 1: 'dog'}
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuración global (constantes) ---
# Se definen aquí para que sean accesibles tanto dentro como fuera del bloque if __name__ == '__main__':
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_NAME = 'pet_classifier.pth'

# Rutas para los directorios del dataset
BASE_DIR = 'dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')

# --- Definición del Modelo (Red Neuronal Convolucional - CNN) ---
# La definición de la clase del modelo DEBE estar fuera del bloque
# if __name__ == '__main__': para que los procesos secundarios (workers) puedan importarla.
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        
        # 1ra Capa Convolucional
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2da Capa Convolucional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3ra Capa Convolucional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4ta Capa Convolucional (cambiado de 128 a 256 filtros como en tu código)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # <- Mantuve 256 como en tu código
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calcular el tamaño de entrada para la primera capa densa (fc1)
        # Después de 4 capas de MaxPool2d (cada una divide por 2), la dimensión espacial se reduce 2^4 = 16 veces.
        # El número de canales de salida de la última capa convolucional (conv4) es 256.
        self.fc1 = nn.Linear(256 * (IMAGE_SIZE[0] // 16) * (IMAGE_SIZE[1] // 16), 512)
        self.dropout = nn.Dropout(0.5)
        
        # Para clasificación binaria con BCELoss, la capa final debe tener 1 salida
        self.fc2 = nn.Linear(512, 1) 
        
    # El método forward debe estar correctamente indentado como un método de la clase
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x)) # Aplicar sigmoide para BCELoss
        return x

# --- Bloque Principal del Programa ---
# Todo el código que se ejecuta directamente debe estar dentro de este bloque
# para evitar problemas de multiprocessing en Windows.
if __name__ == '__main__':
    # Determinar si GPU está disponible
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    # Transformaciones de datos (repetidas aquí para claridad, pero ya estaban bien arriba)
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(40),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Cargar datasets
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=VALIDATION_DIR, transform=val_transforms)

    # Nota importante para Windows:
    # Si el error "RuntimeError: An attempt has been made to start a new process..." persiste,
    # o si quieres simplificar y no usar multiprocessing para la carga de datos,
    # cambia 'num_workers' a 0. Esto hará que la carga sea secuencial en el hilo principal.
    # num_workers=0 es más robusto en Windows.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Índice de clases:", train_dataset.class_to_idx)

    # Instanciar el modelo y enviarlo al dispositivo (CPU/GPU)
    model = CatDogClassifier().to(DEVICE)
    
    # Función de Pérdida y Optimizador
    # BCELoss requiere una salida de 1 neurona con activación sigmoide.
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Inicializar listas para guardar métricas de entrenamiento
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Comenzando entrenamiento...")
    for epoch in range(EPOCHS):
        model.train() # Pone el modelo en modo entrenamiento
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Bucle de entrenamiento
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad() # Reinicia los gradientes acumulados
            
            outputs = model(inputs)
            # labels.float() es necesario porque BCELoss espera tensores flotantes
            # outputs.squeeze() es para asegurar que outputs tenga la misma forma que labels (ej. [batch_size])
            loss = criterion(outputs.squeeze(), labels.float()) 
            loss.backward() # Calcula los gradientes
            optimizer.step() # Actualiza los pesos del modelo
            
            running_loss += loss.item() * inputs.size(0)
            # Para calcular la precisión en clasificación binaria
            # (outputs > 0.5) convierte las probabilidades en 0 o 1
            # .squeeze() ajusta la forma, .long() convierte a tipo entero
            predicted = (outputs > 0.5).squeeze().long() 
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # --- Fase de Evaluación ---
        model.eval() # Pone el modelo en modo evaluación (desactiva dropout, etc.)
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # Desactiva el cálculo de gradientes para la evaluación
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).squeeze().long()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}], '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
    # Guardar el modelo
    # Se guarda solo el 'state_dict' (los pesos aprendidos), no el modelo completo
    # Esto es una buena práctica y hace que el archivo del modelo sea más pequeño.
    torch.save(model.state_dict(), MODEL_NAME)
    print(f'Modelo guardado como {MODEL_NAME}')

    # Visualizar resultados del entrenamiento
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Pérdida por Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Precisión por Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Índice de clases:", train_dataset.class_to_idx)
    print("Entrenamiento completado.")
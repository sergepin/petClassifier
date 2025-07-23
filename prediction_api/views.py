# prediction_api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
import io
import torch
import os

# Importa SOLO la función get_model, preprocess_image y CLASS_NAMES
from .model_loader import get_model, preprocess_image, CLASS_NAMES

# Define el dispositivo (CPU o GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PredictImageView(APIView):
    def post(self, request, *args, **kwargs):
        # Obtener el modelo. Se cargará solo la primera vez que se llame.
        model = get_model()

        if model is None:
            return Response(
                {"error": "El modelo no se ha cargado correctamente. Verifique los logs del servidor para más detalles."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Verifica si se envió un archivo
        if 'image' not in request.FILES:
            return Response(
                {"error": "No se proporcionó ninguna imagen."},
                status=status.HTTP_400_BAD_REQUEST
            )

        image_file = request.FILES['image']

        # Abre la imagen usando PIL (Pillow)
        try:
            # Asegúrate de que la imagen sea un archivo legible
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        except Exception as e:
            return Response(
                {"error": f"Error al abrir la imagen: {e}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Preprocesar la imagen
        # Añade una dimensión de batch (batch_size=1)
        input_tensor = preprocess_image(image).unsqueeze(0) 
        input_tensor = input_tensor.to(DEVICE) # Mueve el tensor al dispositivo

        # Realizar la predicción
        model.eval() # Asegurarse de que el modelo esté en modo evaluación
        with torch.no_grad():
            output = model(input_tensor)
            
        # Convertir la salida del modelo a un valor legible
        # La salida es una única probabilidad (entre 0 y 1)
        # Por diseño de tu modelo (sigmoide en la última capa), esta 'raw_probability_output'
        # representa la probabilidad de la clase con índice 1 (en tu caso, 'dog').
        raw_probability_output = output.item() 

        # Calcular las probabilidades para ambas clases
        prob_dog = raw_probability_output
        prob_cat = 1 - raw_probability_output

        # Determinar la clase predicha y su confianza
        if prob_dog >= 0.5:
            predicted_class_index = 1 # Perro (si 1 es perro)
            confidence = prob_dog # La confianza para el perro es su propia probabilidad
        else:
            predicted_class_index = 0 # Gato (si 0 es gato)
            confidence = prob_cat # La confianza para el gato es 1 - prob_dog (o prob_cat)

        predicted_class_name = CLASS_NAMES.get(predicted_class_index, "desconocido")

        return Response({
            "predicted_class": predicted_class_name,
            "confidence": f"{confidence:.4f}",
            "probabilities": { # <-- NUEVO CAMPO "probabilities" como diccionario
                "cat": f"{prob_cat:.4f}",
                "dog": f"{prob_dog:.4f}"
            }
        }, status=status.HTTP_200_OK)
"""
Configuración del sistema de reconocimiento de estomas
"""
import os

# Rutas
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# Parámetros de imagen
IMG_SIZE = 640  # Para YOLO
UNET_SIZE = (256, 256)  # Para U-Net
BATCH_SIZE = 16

# Parámetros de entrenamiento
EPOCHS = 100
LEARNING_RATE = 0.001
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

# Parámetros de estomas
MIN_STOMATA_AREA = 50  # píxeles mínimos para considerar un estoma
MAX_STOMATA_AREA = 2000  # píxeles máximos

# Configuración de cámara para tiempo real
CAMERA_ID = 0
FPS = 30

# Colores para visualización (BGR)
COLORS = {
    'detected': (0, 255, 0),  # Verde para estomas detectados
    'contour': (255, 0, 0),   # Azul para contornos
    'text': (0, 0, 255)       # Rojo para texto
}
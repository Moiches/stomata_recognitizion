"""
Módulo de preprocesamiento de imágenes para detección de estomas
"""
import cv2
import numpy as np
from typing import Tuple, List
import albumentations as A
from config import UNET_SIZE

class StomataPreprocessor:
    def __init__(self):
        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
        ])
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Mejora el contraste usando CLAHE"""
        if len(image.shape) == 3:
            # Convertir a LAB para mejor mejora de contraste
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        return enhanced
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Reduce el ruido de la imagen"""
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Mejora los bordes para mejor detección de estomas"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar filtro Gaussian para suavizar
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detectar bordes con Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Combinar con imagen original
        if len(image.shape) == 3:
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            enhanced = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
        else:
            enhanced = cv2.addWeighted(image, 0.8, edges, 0.2, 0)
        
        return enhanced
    
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocesa imagen para detección con YOLO"""
        # Redimensionar manteniendo aspecto
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = 640, int(640 * w / h)
        else:
            new_h, new_w = int(640 * h / w), 640
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Crear canvas de 640x640 y centrar imagen
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        start_y = (640 - new_h) // 2
        start_x = (640 - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        # Mejorar contraste
        enhanced = self.enhance_contrast(canvas)
        
        # Reducir ruido
        denoised = self.denoise_image(enhanced)
        
        return denoised
    
    def preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Preprocesa imagen para segmentación con U-Net"""
        # Redimensionar a tamaño fijo para U-Net
        resized = cv2.resize(image, UNET_SIZE)
        
        # Mejorar contraste
        enhanced = self.enhance_contrast(resized)
        
        # Normalizar valores de píxeles
        normalized = enhanced.astype(np.float32) / 255.0
        
        return normalized
    
    def augment_image(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica aumentaciones de datos"""
        if mask is not None:
            augmented = self.augmentations(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.augmentations(image=image)
            return augmented['image'], None
    
    def create_binary_mask(self, image: np.ndarray, threshold_method='adaptive') -> np.ndarray:
        """Crea máscara binaria para detectar estructuras estomatales"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if threshold_method == 'adaptive':
            # Umbralización adaptiva
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        elif threshold_method == 'otsu':
            # Método de Otsu
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            # Umbralización fija
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Operaciones morfológicas para limpiar la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
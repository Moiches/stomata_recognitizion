"""
Segmentador de estomas usando U-Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List, Dict
import segmentation_models_pytorch as smp
from config import UNET_SIZE
import os

class UNetStomataSegmenter:
    def __init__(self, 
                 encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet',
                 classes: int = 2,  # fondo + estomas
                 activation: str = 'sigmoid'):
        """
        Inicializa el segmentador U-Net
        Args:
            encoder_name: Nombre del encoder (resnet34, efficientnet-b0, etc.)
            encoder_weights: Pesos preentrenados
            classes: Número de clases (fondo + estomas)
            activation: Función de activación
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Crear modelo U-Net
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        )
        
        self.model.to(self.device)
        self.classes = classes
        
        # Configurar loss y optimizer
        self.criterion = smp.losses.DiceLoss(mode='binary' if classes == 2 else 'multiclass')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocesa imagen para el modelo
        Args:
            image: Imagen de entrada
        Returns:
            Tensor preparado para el modelo
        """
        # Redimensionar
        resized = cv2.resize(image, UNET_SIZE)
        
        # Normalizar
        normalized = resized.astype(np.float32) / 255.0
        
        # Convertir a tensor y añadir dimensiones de batch
        if len(normalized.shape) == 3:
            tensor = torch.from_numpy(normalized).permute(2, 0, 1)  # HWC -> CHW
        else:
            tensor = torch.from_numpy(normalized).unsqueeze(0)  # Añadir canal
        
        tensor = tensor.unsqueeze(0)  # Añadir batch dimension
        return tensor.to(self.device)
    
    def postprocess_mask(self, mask: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocesa la máscara de segmentación
        Args:
            mask: Máscara predicha por el modelo
            original_size: Tamaño original de la imagen (H, W)
        Returns:
            Máscara binaria redimensionada
        """
        # Convertir a numpy y quitar dimensiones extra
        if self.classes == 2:
            mask_np = torch.sigmoid(mask).cpu().numpy().squeeze()
            # Umbralizar
            binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
        else:
            mask_np = torch.softmax(mask, dim=1).cpu().numpy().squeeze()
            # Tomar clase con mayor probabilidad
            binary_mask = np.argmax(mask_np, axis=0).astype(np.uint8) * 255
        
        # Redimensionar al tamaño original
        resized_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))
        
        return resized_mask
    
    def segment_stomata(self, image: np.ndarray) -> np.ndarray:
        """
        Segmenta estomas en una imagen
        Args:
            image: Imagen de entrada
        Returns:
            Máscara binaria con estomas segmentados
        """
        original_size = image.shape[:2]
        
        # Preprocesar
        input_tensor = self.preprocess_image(image)
        
        # Inferencia
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Postprocesar
        mask = self.postprocess_mask(prediction, original_size)
        
        return mask
    
    def extract_stomata_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Extrae contornos de estomas de la máscara
        Args:
            mask: Máscara binaria
        Returns:
            Lista de contornos
        """
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar contornos por área
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # Filtrar por tamaño típico de estomas
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def analyze_stomata_shape(self, contour: np.ndarray) -> Dict:
        """
        Analiza la forma de un estoma individual
        Args:
            contour: Contorno del estoma
        Returns:
            Diccionario con características del estoma
        """
        # Área
        area = cv2.contourArea(contour)
        
        # Perímetro
        perimeter = cv2.arcLength(contour, True)
        
        # Centro de masa
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        # Elipse que mejor se ajusta
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        else:
            major_axis = minor_axis = eccentricity = angle = 0
        
        # Rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contour)
        
        # Circularidad (4π*área/perímetro²)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Relación de aspecto
        aspect_ratio = w / h if h > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'center': (cx, cy),
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'eccentricity': eccentricity,
            'angle': angle,
            'bbox': (x, y, w, h),
            'circularity': circularity,
            'aspect_ratio': aspect_ratio
        }
    
    def draw_segmentation(self, 
                         image: np.ndarray, 
                         mask: np.ndarray, 
                         contours: List[np.ndarray] = None,
                         show_analysis: bool = True) -> np.ndarray:
        """
        Dibuja los resultados de segmentación
        Args:
            image: Imagen original
            mask: Máscara de segmentación
            contours: Lista de contornos (opcional)
            show_analysis: Mostrar análisis de forma
        Returns:
            Imagen con segmentación dibujada
        """
        result = image.copy()
        
        # Superponer máscara
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        result = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)
        
        if contours is not None:
            for i, contour in enumerate(contours):
                # Dibujar contorno
                cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                
                if show_analysis:
                    # Analizar forma
                    analysis = self.analyze_stomata_shape(contour)
                    
                    # Dibujar centro
                    cv2.circle(result, analysis['center'], 3, (255, 0, 0), -1)
                    
                    # Etiqueta con información
                    label = f"#{i+1} A:{analysis['area']:.0f}"
                    cv2.putText(result, label, 
                              (analysis['center'][0] + 10, analysis['center'][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result
    
    def train_step(self, images: torch.Tensor, masks: torch.Tensor) -> float:
        """
        Paso de entrenamiento
        Args:
            images: Batch de imágenes
            masks: Batch de máscaras
        Returns:
            Pérdida del paso
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(images)
        loss = self.criterion(predictions, masks)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str):
        """Guarda el modelo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Carga el modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
"""
Detector de estomas usando YOLO
"""
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict
from config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MODELS_DIR
import os

class StomataYOLODetector:
    def __init__(self, model_path: str = None):
        """
        Inicializa el detector YOLO
        Args:
            model_path: Ruta al modelo YOLO entrenado. Si es None, usa YOLOv8n
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Usar modelo preentrenado YOLOv8n como base
            self.model = YOLO('yolov8n.pt')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def train_model(self, 
                   data_yaml_path: str, 
                   epochs: int = 100, 
                   batch_size: int = 16,
                   img_size: int = 640):
        """
        Entrena el modelo YOLO con datos de estomas
        Args:
            data_yaml_path: Ruta al archivo YAML con configuración de datos
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del lote
            img_size: Tamaño de imagen
        """
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            save=True,
            project=MODELS_DIR,
            name='stomata_yolo'
        )
        return results
    
    def detect_stomata(self, 
                      image: np.ndarray, 
                      confidence: float = CONFIDENCE_THRESHOLD) -> List[Dict]:
        """
        Detecta estomas en una imagen
        Args:
            image: Imagen de entrada
            confidence: Umbral de confianza
        Returns:
            Lista de detecciones con información de cada estoma
        """
        # Realizar predicción
        results = self.model(image, conf=confidence, iou=IOU_THRESHOLD)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Coordenadas del bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence_score = boxes.conf[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence_score),
                    'class_id': class_id,
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    'area': int((x2 - x1) * (y2 - y1))
                }
                detections.append(detection)
        
        return detections
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Dict],
                       show_confidence: bool = True) -> np.ndarray:
        """
        Dibuja las detecciones en la imagen
        Args:
            image: Imagen original
            detections: Lista de detecciones
            show_confidence: Mostrar valor de confianza
        Returns:
            Imagen con detecciones dibujadas
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Dibujar bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar punto central
            center_x, center_y = detection['center']
            cv2.circle(result_image, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Etiqueta con confianza
            if show_confidence:
                label = f'Estoma: {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(result_image, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            (0, 255, 0), -1)
                cv2.putText(result_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result_image
    
    def count_stomata(self, detections: List[Dict]) -> Dict:
        """
        Cuenta y analiza los estomas detectados
        Args:
            detections: Lista de detecciones
        Returns:
            Diccionario con estadísticas
        """
        total_count = len(detections)
        
        if total_count == 0:
            return {
                'total_count': 0,
                'avg_confidence': 0,
                'avg_area': 0,
                'min_area': 0,
                'max_area': 0
            }
        
        confidences = [d['confidence'] for d in detections]
        areas = [d['area'] for d in detections]
        
        stats = {
            'total_count': total_count,
            'avg_confidence': np.mean(confidences),
            'avg_area': np.mean(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'std_area': np.std(areas)
        }
        
        return stats
    
    def calculate_stomatal_density(self, 
                                  detections: List[Dict], 
                                  image_area_mm2: float) -> float:
        """
        Calcula la densidad estomatal
        Args:
            detections: Lista de detecciones
            image_area_mm2: Área de la imagen en mm²
        Returns:
            Densidad estomatal (estomas/mm²)
        """
        stomata_count = len(detections)
        if image_area_mm2 <= 0:
            return 0.0
        
        density = stomata_count / image_area_mm2
        return density
    
    def save_model(self, save_path: str):
        """Guarda el modelo entrenado"""
        self.model.save(save_path)
    
    def load_model(self, model_path: str):
        """Carga un modelo entrenado"""
        self.model = YOLO(model_path)
        self.model.to(self.device)
"""
Analizador completo de estomas que combina detección, segmentación y cálculo de densidad
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os

from yolo_detector import StomataYOLODetector
from unet_segmentation import UNetStomataSegmenter
from preprocessor import StomataPreprocessor

@dataclass
class StomataAnalysis:
    """Clase para almacenar resultados del análisis de estomas"""
    total_count: int
    stomatal_density: float  # estomas/mm²
    avg_area: float
    avg_circularity: float
    detections: List[Dict]
    contours: List[np.ndarray]
    statistics: Dict
    image_area_mm2: float

class StomataAnalyzer:
    def __init__(self, 
                 yolo_model_path: str = None,
                 unet_model_path: str = None,
                 use_yolo: bool = True,
                 use_unet: bool = True):
        """
        Inicializa el analizador completo de estomas
        Args:
            yolo_model_path: Ruta al modelo YOLO entrenado
            unet_model_path: Ruta al modelo U-Net entrenado
            use_yolo: Usar detección YOLO
            use_unet: Usar segmentación U-Net
        """
        self.preprocessor = StomataPreprocessor()
        
        self.use_yolo = use_yolo
        self.use_unet = use_unet
        
        if use_yolo:
            self.yolo_detector = StomataYOLODetector(yolo_model_path)
        
        if use_unet:
            self.unet_segmenter = UNetStomataSegmenter()
            if unet_model_path and os.path.exists(unet_model_path):
                self.unet_segmenter.load_model(unet_model_path)
    
    def calculate_image_area(self, 
                           image: np.ndarray, 
                           scale_factor: float = 1.0,
                           pixel_size_um: float = 1.0) -> float:
        """
        Calcula el área real de la imagen en mm²
        Args:
            image: Imagen de entrada
            scale_factor: Factor de escala si es conocido
            pixel_size_um: Tamaño del píxel en micrómetros
        Returns:
            Área en mm²
        """
        height, width = image.shape[:2]
        
        # Convertir píxeles a micrómetros y luego a mm²
        area_pixels = height * width
        area_um2 = area_pixels * (pixel_size_um ** 2) * (scale_factor ** 2)
        area_mm2 = area_um2 / 1_000_000  # convertir µm² a mm²
        
        return area_mm2
    
    def analyze_image(self, 
                     image: np.ndarray,
                     pixel_size_um: float = 1.0,
                     scale_factor: float = 1.0,
                     confidence_threshold: float = 0.5) -> StomataAnalysis:
        """
        Análisis completo de estomas en una imagen
        Args:
            image: Imagen de entrada
            pixel_size_um: Tamaño del píxel en micrómetros
            scale_factor: Factor de escala
            confidence_threshold: Umbral de confianza para detección
        Returns:
            Análisis completo de estomas
        """
        # Calcular área real de la imagen
        image_area_mm2 = self.calculate_image_area(image, scale_factor, pixel_size_um)
        
        detections = []
        contours = []
        
        # Preprocesar imagen
        preprocessed = self.preprocessor.preprocess_for_detection(image)
        
        # Detección con YOLO
        if self.use_yolo:
            yolo_detections = self.yolo_detector.detect_stomata(
                preprocessed, confidence_threshold
            )
            detections.extend(yolo_detections)
        
        # Segmentación con U-Net
        if self.use_unet:
            mask = self.unet_segmenter.segment_stomata(image)
            unet_contours = self.unet_segmenter.extract_stomata_contours(mask)
            contours.extend(unet_contours)
            
            # Convertir contornos a detecciones para unificar análisis
            for contour in unet_contours:
                analysis = self.unet_segmenter.analyze_stomata_shape(contour)
                x, y, w, h = analysis['bbox']
                detection = {
                    'bbox': [x, y, x+w, y+h],
                    'center': analysis['center'],
                    'area': analysis['area'],
                    'confidence': 0.9,  # Confianza alta para segmentación
                    'circularity': analysis['circularity'],
                    'aspect_ratio': analysis['aspect_ratio'],
                    'source': 'unet'
                }
                detections.append(detection)
        
        # Fusionar detecciones superpuestas (opcional)
        if self.use_yolo and self.use_unet:
            detections = self._merge_overlapping_detections(detections)
        
        # Calcular estadísticas
        statistics = self._calculate_statistics(detections, contours)
        
        # Calcular densidad estomatal
        total_count = len(detections)
        stomatal_density = total_count / image_area_mm2 if image_area_mm2 > 0 else 0
        
        # Calcular promedios
        avg_area = np.mean([d['area'] for d in detections]) if detections else 0
        avg_circularity = np.mean([d.get('circularity', 0) for d in detections]) if detections else 0
        
        return StomataAnalysis(
            total_count=total_count,
            stomatal_density=stomatal_density,
            avg_area=avg_area,
            avg_circularity=avg_circularity,
            detections=detections,
            contours=contours,
            statistics=statistics,
            image_area_mm2=image_area_mm2
        )
    
    def _merge_overlapping_detections(self, 
                                    detections: List[Dict], 
                                    iou_threshold: float = 0.5) -> List[Dict]:
        """
        Fusiona detecciones superpuestas de YOLO y U-Net
        Args:
            detections: Lista de detecciones
            iou_threshold: Umbral de IoU para considerar superposición
        Returns:
            Lista de detecciones filtradas
        """
        if not detections:
            return detections
        
        # Separar detecciones por fuente
        yolo_detections = [d for d in detections if d.get('source') != 'unet']
        unet_detections = [d for d in detections if d.get('source') == 'unet']
        
        merged = []
        used_unet_indices = set()
        
        # Para cada detección YOLO, buscar correspondencia en U-Net
        for yolo_det in yolo_detections:
            best_match = None
            best_iou = 0
            best_idx = -1
            
            for i, unet_det in enumerate(unet_detections):
                if i in used_unet_indices:
                    continue
                
                iou = self._calculate_iou(yolo_det['bbox'], unet_det['bbox'])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_match = unet_det
                    best_idx = i
            
            if best_match:
                # Fusionar información de ambas detecciones
                merged_det = yolo_det.copy()
                merged_det.update({
                    'circularity': best_match.get('circularity', 0),
                    'aspect_ratio': best_match.get('aspect_ratio', 0),
                    'unet_confidence': best_match.get('confidence', 0),
                    'source': 'merged'
                })
                merged.append(merged_det)
                used_unet_indices.add(best_idx)
            else:
                merged.append(yolo_det)
        
        # Añadir detecciones U-Net sin correspondencia
        for i, unet_det in enumerate(unet_detections):
            if i not in used_unet_indices:
                merged.append(unet_det)
        
        return merged
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calcula Intersection over Union entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_statistics(self, 
                            detections: List[Dict], 
                            contours: List[np.ndarray]) -> Dict:
        """Calcula estadísticas detalladas"""
        if not detections:
            return {
                'count': 0,
                'areas': [],
                'confidences': [],
                'size_distribution': {}
            }
        
        areas = [d['area'] for d in detections]
        confidences = [d['confidence'] for d in detections]
        
        # Distribución por tamaños
        size_distribution = {
            'small': len([a for a in areas if a < 200]),
            'medium': len([a for a in areas if 200 <= a < 500]),
            'large': len([a for a in areas if a >= 500])
        }
        
        return {
            'count': len(detections),
            'areas': areas,
            'confidences': confidences,
            'area_stats': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'size_distribution': size_distribution
        }
    
    def visualize_analysis(self, 
                          image: np.ndarray, 
                          analysis: StomataAnalysis,
                          show_detections: bool = True,
                          show_contours: bool = True,
                          show_stats: bool = True) -> np.ndarray:
        """
        Visualiza los resultados del análisis
        Args:
            image: Imagen original
            analysis: Resultado del análisis
            show_detections: Mostrar detecciones YOLO
            show_contours: Mostrar contornos U-Net
            show_stats: Mostrar estadísticas
        Returns:
            Imagen con visualización
        """
        result = image.copy()
        
        # Dibujar detecciones YOLO
        if show_detections and self.use_yolo:
            result = self.yolo_detector.draw_detections(result, analysis.detections)
        
        # Dibujar contornos U-Net
        if show_contours and self.use_unet and analysis.contours:
            for i, contour in enumerate(analysis.contours):
                cv2.drawContours(result, [contour], -1, (255, 0, 255), 2)
        
        # Añadir estadísticas
        if show_stats:
            stats_text = [
                f"Total estomas: {analysis.total_count}",
                f"Densidad: {analysis.stomatal_density:.2f} est/mm²",
                f"Área promedio: {analysis.avg_area:.1f} px²",
                f"Área imagen: {analysis.image_area_mm2:.3f} mm²"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(result, text, (10, 30 + i * 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result, text, (10, 30 + i * 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return result
    
    def save_analysis(self, analysis: StomataAnalysis, output_path: str):
        """Guarda el análisis en formato JSON"""
        # Convertir contours a lista serializable
        serializable_analysis = {
            'total_count': analysis.total_count,
            'stomatal_density': analysis.stomatal_density,
            'avg_area': analysis.avg_area,
            'avg_circularity': analysis.avg_circularity,
            'image_area_mm2': analysis.image_area_mm2,
            'detections': analysis.detections,
            'statistics': analysis.statistics,
            'contour_count': len(analysis.contours)
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
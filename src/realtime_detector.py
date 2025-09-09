"""
Sistema de reconocimiento de estomas en tiempo real usando cámara
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable
from queue import Queue
import tkinter as tk
from tkinter import ttk

from stomata_analyzer import StomataAnalyzer
from config import CAMERA_ID, FPS

class RealtimeStomataDetector:
    def __init__(self,
                 yolo_model_path: str = None,
                 unet_model_path: str = None,
                 camera_id: int = CAMERA_ID,
                 pixel_size_um: float = 1.0,
                 scale_factor: float = 1.0):
        """
        Inicializa el detector en tiempo real
        Args:
            yolo_model_path: Ruta al modelo YOLO entrenado
            unet_model_path: Ruta al modelo U-Net entrenado
            camera_id: ID de la cámara
            pixel_size_um: Tamaño del píxel en micrómetros
            scale_factor: Factor de escala
        """
        self.analyzer = StomataAnalyzer(yolo_model_path, unet_model_path)
        self.camera_id = camera_id
        self.pixel_size_um = pixel_size_um
        self.scale_factor = scale_factor
        
        # Control de video
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.frame_count = 0
        
        # Threading
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.analysis_thread = None
        
        # Estadísticas
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # Configuración
        self.confidence_threshold = 0.5
        self.show_detections = True
        self.show_segmentation = True
        self.show_stats = True
        
        # Callback para actualizaciones
        self.update_callback: Optional[Callable] = None
    
    def start_camera(self) -> bool:
        """
        Inicia la cámara
        Returns:
            True si se inició correctamente
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Error: No se pudo abrir la cámara {self.camera_id}")
                return False
            
            # Configurar cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
            
            print(f"Cámara iniciada: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {self.cap.get(cv2.CAP_PROP_FPS)} FPS")
            return True
            
        except Exception as e:
            print(f"Error iniciando cámara: {e}")
            return False
    
    def stop_camera(self):
        """Detiene la cámara"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def start_detection(self):
        """Inicia la detección en tiempo real"""
        if not self.cap or not self.cap.isOpened():
            print("La cámara no está iniciada")
            return
        
        self.is_running = True
        
        # Iniciar hilo de análisis
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()
        
        print("Detección en tiempo real iniciada")
    
    def stop_detection(self):
        """Detiene la detección"""
        self.is_running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1)
        print("Detección detenida")
    
    def _analysis_worker(self):
        """Trabajador de análisis que procesa frames en segundo plano"""
        while self.is_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                try:
                    # Realizar análisis
                    analysis = self.analyzer.analyze_image(
                        frame,
                        pixel_size_um=self.pixel_size_um,
                        scale_factor=self.scale_factor,
                        confidence_threshold=self.confidence_threshold
                    )
                    
                    # Crear visualización
                    visualization = self.analyzer.visualize_analysis(
                        frame, analysis,
                        show_detections=self.show_detections,
                        show_contours=self.show_segmentation,
                        show_stats=self.show_stats
                    )
                    
                    # Añadir información adicional
                    self._add_realtime_info(visualization, analysis)
                    
                    # Enviar resultado
                    if not self.result_queue.full():
                        self.result_queue.put((visualization, analysis))
                    
                except Exception as e:
                    print(f"Error en análisis: {e}")
                    continue
            else:
                time.sleep(0.001)  # Pequeña pausa para evitar uso excesivo de CPU
    
    def _add_realtime_info(self, image: np.ndarray, analysis):
        """Añade información adicional para tiempo real"""
        height, width = image.shape[:2]
        
        # FPS
        cv2.putText(image, f"FPS: {self.current_fps:.1f}", 
                   (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame count
        cv2.putText(image, f"Frame: {self.frame_count}", 
                   (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Estado de grabación
        if self.is_recording:
            cv2.circle(image, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(image, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def get_next_frame(self) -> Optional[tuple]:
        """
        Obtiene el siguiente frame procesado
        Returns:
            Tuple (imagen_procesada, análisis) o None
        """
        if not self.cap or not self.is_running:
            return None
        
        # Capturar frame
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        
        # Calcular FPS
        self.fps_counter += 1
        if time.time() - self.fps_timer > 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = time.time()
        
        # Enviar frame para análisis
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())
        
        # Obtener resultado procesado
        if not self.result_queue.empty():
            return self.result_queue.get()
        else:
            # Si no hay resultado procesado, mostrar frame original con info básica
            display_frame = frame.copy()
            cv2.putText(display_frame, "Procesando...", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", 
                       (display_frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return display_frame, None
    
    def set_confidence_threshold(self, threshold: float):
        """Establece el umbral de confianza"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
    
    def toggle_detections(self):
        """Activa/desactiva mostrar detecciones"""
        self.show_detections = not self.show_detections
    
    def toggle_segmentation(self):
        """Activa/desactiva mostrar segmentación"""
        self.show_segmentation = not self.show_segmentation
    
    def toggle_stats(self):
        """Activa/desactiva mostrar estadísticas"""
        self.show_stats = not self.show_stats
    
    def start_recording(self, output_path: str = "stomata_recording.avi"):
        """Inicia grabación de video"""
        if not self.cap:
            return False
        
        # Configurar writer de video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, FPS, (frame_width, frame_height)
        )
        
        self.is_recording = True
        print(f"Iniciando grabación: {output_path}")
        return True
    
    def stop_recording(self):
        """Detiene la grabación"""
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        print("Grabación detenida")
    
    def save_snapshot(self, filename: str = None):
        """Guarda una captura actual"""
        if not self.cap:
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        if filename is None:
            filename = f"snapshot_{int(time.time())}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"Captura guardada: {filename}")
        return True
    
    def run_window_display(self):
        """Ejecuta la visualización en ventana de OpenCV"""
        if not self.start_camera():
            return
        
        self.start_detection()
        
        print("Controles:")
        print("  q - Salir")
        print("  d - Activar/desactivar detecciones")
        print("  s - Activar/desactivar segmentación")
        print("  i - Activar/desactivar información")
        print("  r - Iniciar/detener grabación")
        print("  c - Captura")
        print("  + - Aumentar confianza")
        print("  - - Disminuir confianza")
        
        while True:
            result = self.get_next_frame()
            if result is None:
                break
            
            display_frame, analysis = result
            
            # Mostrar frame
            cv2.imshow("Detección de Estomas en Tiempo Real", display_frame)
            
            # Grabar si está activado
            if self.is_recording and hasattr(self, 'video_writer') and self.video_writer:
                self.video_writer.write(display_frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.toggle_detections()
            elif key == ord('s'):
                self.toggle_segmentation()
            elif key == ord('i'):
                self.toggle_stats()
            elif key == ord('r'):
                if self.is_recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key == ord('c'):
                self.save_snapshot()
            elif key == ord('+') or key == ord('='):
                self.set_confidence_threshold(self.confidence_threshold + 0.05)
                print(f"Confianza: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.set_confidence_threshold(self.confidence_threshold - 0.05)
                print(f"Confianza: {self.confidence_threshold:.2f}")
        
        # Limpieza
        self.stop_detection()
        self.stop_recording()
        self.stop_camera()
        cv2.destroyAllWindows()
    
    def set_update_callback(self, callback: Callable):
        """Establece callback para actualizaciones"""
        self.update_callback = callback
    
    def cleanup(self):
        """Limpieza completa del detector"""
        self.stop_detection()
        self.stop_recording()
        self.stop_camera()
        cv2.destroyAllWindows()

# Función de conveniencia para ejecutar directamente
def main():
    detector = RealtimeStomataDetector()
    try:
        detector.run_window_display()
    except KeyboardInterrupt:
        print("\\nDeteniendo detector...")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()
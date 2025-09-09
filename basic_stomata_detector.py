"""
Versión básica del reconocedor de estomas para pruebas iniciales
Solo requiere OpenCV, NumPy y Ultralytics
"""
import cv2
import numpy as np
import os

class BasicStomataDetector:
    def __init__(self):
        """Detector básico usando solo OpenCV y técnicas clásicas"""
        self.min_area = 50
        self.max_area = 2000
        
    def preprocess_image(self, image):
        """Preprocesamiento básico"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Reducir ruido
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def detect_stomata_basic(self, image):
        """Detección básica usando técnicas de OpenCV"""
        preprocessed = self.preprocess_image(image)
        
        # Umbralización adaptiva
        binary = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar por área
        stomata_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                stomata_contours.append(contour)
        
        return stomata_contours
    
    def analyze_contours(self, contours):
        """Análisis básico de contornos"""
        results = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Centro de masa
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0
            
            # Circularidad
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            results.append({
                'id': i,
                'center': (cx, cy),
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity
            })
        
        return results
    
    def visualize_results(self, image, contours, analysis):
        """Visualizar resultados"""
        result_image = image.copy()
        
        for i, (contour, info) in enumerate(zip(contours, analysis)):
            # Dibujar contorno
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
            
            # Marcar centro
            cv2.circle(result_image, info['center'], 3, (255, 0, 0), -1)
            
            # Etiqueta
            label = f"#{i+1}"
            cv2.putText(result_image, label, 
                       (info['center'][0] + 10, info['center'][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Información general
        info_text = f"Total estomas detectados: {len(contours)}"
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return result_image
    
    def process_image_file(self, image_path, save_result=True):
        """Procesar archivo de imagen"""
        if not os.path.exists(image_path):
            print(f"Error: No se encontró la imagen {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar {image_path}")
            return None
        
        print(f"Procesando: {os.path.basename(image_path)}")
        
        # Detectar estomas
        contours = self.detect_stomata_basic(image)
        
        # Analizar resultados
        analysis = self.analyze_contours(contours)
        
        # Calcular estadísticas
        if analysis:
            areas = [a['area'] for a in analysis]
            circularities = [a['circularity'] for a in analysis]
            
            stats = {
                'total_stomata': len(contours),
                'avg_area': np.mean(areas),
                'std_area': np.std(areas),
                'avg_circularity': np.mean(circularities),
                'area_range': (np.min(areas), np.max(areas))
            }
        else:
            stats = {'total_stomata': 0}
        
        print(f"Resultados:")
        print(f"  - Total estomas: {stats['total_stomata']}")
        if stats['total_stomata'] > 0:
            print(f"  - Área promedio: {stats['avg_area']:.1f} px²")
            print(f"  - Circularidad promedio: {stats['avg_circularity']:.3f}")
        
        # Crear visualización
        result_image = self.visualize_results(image, contours, analysis)
        
        # Guardar resultado
        if save_result:
            output_path = f"basic_result_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, result_image)
            print(f"  - Resultado guardado: {output_path}")
        
        return {
            'stats': stats,
            'contours': contours,
            'analysis': analysis,
            'result_image': result_image
        }

def test_basic_detector():
    """Función de prueba"""
    detector = BasicStomataDetector()
    
    # Crear imagen de prueba si no hay imágenes reales
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 128
    
    # Simular algunos "estomas" como círculos
    cv2.circle(test_image, (150, 150), 30, (80, 80, 80), -1)
    cv2.circle(test_image, (300, 200), 25, (90, 90, 90), -1)
    cv2.circle(test_image, (450, 180), 35, (70, 70, 70), -1)
    cv2.circle(test_image, (200, 300), 28, (85, 85, 85), -1)
    
    # Añadir algo de ruido
    noise = np.random.normal(0, 10, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Guardar imagen de prueba
    cv2.imwrite("test_stomata_image.jpg", test_image)
    print("✅ Imagen de prueba creada: test_stomata_image.jpg")
    
    # Procesar
    result = detector.process_image_file("test_stomata_image.jpg")
    
    if result:
        print("✅ Detector básico funcionando correctamente!")
        return True
    else:
        print("❌ Error en el detector básico")
        return False

if __name__ == "__main__":
    print("🌿 Probando Detector Básico de Estomas")
    print("=" * 40)
    
    # Verificar dependencias mínimas
    try:
        import cv2
        import numpy as np
        print("✅ OpenCV y NumPy disponibles")
    except ImportError as e:
        print(f"❌ Error de dependencias: {e}")
        print("Instalar con: pip install opencv-python numpy")
        exit(1)
    
    # Probar detector
    test_basic_detector()
    
    print("\nPara usar con tus propias imágenes:")
    print("detector = BasicStomataDetector()")
    print("result = detector.process_image_file('tu_imagen.jpg')")
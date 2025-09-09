"""
Ejemplos de uso del sistema de reconocimiento de estomas
"""
import sys
import os
import cv2
import numpy as np

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stomata_analyzer import StomataAnalyzer
from batch_processor import BatchStomataProcessor
from realtime_detector import RealtimeStomataDetector

def ejemplo_analisis_individual():
    """Ejemplo de análisis de una sola imagen"""
    print("=== EJEMPLO 1: Análisis de Imagen Individual ===")
    
    # Crear analizador
    analyzer = StomataAnalyzer()
    
    # Simular carga de imagen (reemplazar con ruta real)
    # image = cv2.imread("path/to/your/stomata_image.jpg")
    
    # Para este ejemplo, crear una imagen sintética
    image = np.ones((600, 800, 3), dtype=np.uint8) * 128  # Imagen gris
    
    # Configurar parámetros
    pixel_size_um = 0.5  # 0.5 micrómetros por píxel
    scale_factor = 1.0    # Sin factor de escala adicional
    confidence = 0.6      # Umbral de confianza del 60%
    
    print(f"Configuración:")
    print(f"- Tamaño de píxel: {pixel_size_um} μm")
    print(f"- Factor de escala: {scale_factor}")
    print(f"- Umbral de confianza: {confidence}")
    
    # Realizar análisis
    try:
        analysis = analyzer.analyze_image(
            image,
            pixel_size_um=pixel_size_um,
            scale_factor=scale_factor,
            confidence_threshold=confidence
        )
        
        # Mostrar resultados
        print(f"\\nResultados:")
        print(f"- Total de estomas: {analysis.total_count}")
        print(f"- Densidad estomatal: {analysis.stomatal_density:.2f} estomas/mm²")
        print(f"- Área promedio: {analysis.avg_area:.1f} píxeles²")
        print(f"- Circularidad promedio: {analysis.avg_circularity:.3f}")
        print(f"- Área de imagen: {analysis.image_area_mm2:.3f} mm²")
        
        # Crear visualización
        visualization = analyzer.visualize_analysis(image, analysis)
        
        # Guardar resultado (opcional)
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/ejemplo_individual.jpg", visualization)
        print(f"\\nVisualización guardada en: output/ejemplo_individual.jpg")
        
        # Guardar análisis en JSON
        analyzer.save_analysis(analysis, "output/ejemplo_individual.json")
        print(f"Análisis guardado en: output/ejemplo_individual.json")
        
    except Exception as e:
        print(f"Error en el análisis: {e}")

def ejemplo_procesamiento_lote():
    """Ejemplo de procesamiento en lote"""
    print("\\n=== EJEMPLO 2: Procesamiento en Lote ===")
    
    # Crear procesador en lote
    processor = BatchStomataProcessor(
        output_dir="output",
        pixel_size_um=0.5
    )
    
    print("Configuración del lote:")
    print("- Directorio de salida: output/")
    print("- Tamaño de píxel: 0.5 μm")
    
    # Simular directorio con imágenes
    # En uso real, especificar ruta real: processor.process_image_batch("path/to/images/")
    
    print("\\nEn uso real:")
    print("results = processor.process_image_batch(")
    print("    image_directory='path/to/images/',")
    print("    scale_factor=1.0,")
    print("    confidence_threshold=0.5,")
    print("    save_visualizations=True,")
    print("    save_individual_reports=True")
    print(")")
    print("\\nExportar resultados:")
    print("processor.export_to_csv(results['results'])")
    print("processor.export_to_excel(results['results'], results['consolidated_report'])")

def ejemplo_tiempo_real():
    """Ejemplo de configuración para tiempo real"""
    print("\\n=== EJEMPLO 3: Detección en Tiempo Real ===")
    
    print("Configuración para tiempo real:")
    print("- Requiere cámara web conectada")
    print("- Detección en vivo con visualización")
    print("- Controles interactivos")
    
    # Crear detector (sin iniciar automáticamente)
    detector = RealtimeStomataDetector(
        camera_id=0,
        pixel_size_um=0.5,
        scale_factor=1.0
    )
    
    print("\\nPara ejecutar tiempo real:")
    print("detector = RealtimeStomataDetector()")
    print("detector.run_window_display()")
    print("\\nControles:")
    print("- q: Salir")
    print("- d: Activar/desactivar detecciones") 
    print("- s: Activar/desactivar segmentación")
    print("- r: Iniciar/detener grabación")
    print("- c: Tomar captura")
    print("- +/-: Ajustar confianza")
    
    # No ejecutar automáticamente para evitar bloqueos
    print("\\n(Detector configurado pero no ejecutado automáticamente)")

def ejemplo_configuracion_avanzada():
    """Ejemplo de configuración avanzada"""
    print("\\n=== EJEMPLO 4: Configuración Avanzada ===")
    
    # Configurar analizador con opciones específicas
    analyzer = StomataAnalyzer(
        yolo_model_path=None,  # Usar modelo por defecto
        unet_model_path=None,  # Usar modelo por defecto  
        use_yolo=True,         # Activar YOLO
        use_unet=False         # Desactivar U-Net para este ejemplo
    )
    
    print("Configuración del analizador:")
    print("- YOLO: Activado")
    print("- U-Net: Desactivado")
    print("- Modelos: Por defecto")
    
    # Ejemplo de calibración precisa
    pixel_size_um = 0.25  # Microscopio de alta resolución
    scale_factor = 1.5    # Factor de magnificación adicional
    
    # Calcular área real de muestra
    image_width_px = 1920
    image_height_px = 1080
    
    area_real_mm2 = (image_width_px * pixel_size_um / 1000) * (image_height_px * pixel_size_um / 1000)
    area_real_mm2 *= (scale_factor ** 2)
    
    print(f"\\nCálculos de calibración:")
    print(f"- Resolución: {image_width_px}x{image_height_px} px")
    print(f"- Tamaño píxel: {pixel_size_um} μm")
    print(f"- Factor escala: {scale_factor}")
    print(f"- Área real calculada: {area_real_mm2:.4f} mm²")
    
    # Parámetros de análisis específicos
    config_params = {
        'confidence_threshold': 0.7,      # Alta confianza
        'min_stomata_area': 100,          # Área mínima en píxeles
        'max_stomata_area': 1500,         # Área máxima en píxeles
        'circularity_threshold': 0.3      # Umbral de circularidad mínima
    }
    
    print(f"\\nParámetros de análisis:")
    for param, value in config_params.items():
        print(f"- {param}: {value}")

def ejemplo_analisis_estadistico():
    """Ejemplo de análisis estadístico de resultados"""
    print("\\n=== EJEMPLO 5: Análisis Estadístico ===")
    
    # Simular resultados de múltiples imágenes
    resultados_simulados = [
        {'stomata_count': 45, 'density': 125.3, 'avg_area': 220.5},
        {'stomata_count': 38, 'density': 108.7, 'avg_area': 195.2},
        {'stomata_count': 52, 'density': 143.1, 'avg_area': 240.8},
        {'stomata_count': 41, 'density': 117.9, 'avg_area': 210.3},
        {'stomata_count': 47, 'density': 132.4, 'avg_area': 225.7}
    ]
    
    # Calcular estadísticas
    stomata_counts = [r['stomata_count'] for r in resultados_simulados]
    densities = [r['density'] for r in resultados_simulados]
    avg_areas = [r['avg_area'] for r in resultados_simulados]
    
    print("Resultados de análisis estadístico:")
    print(f"\\nConteo de estomas:")
    print(f"- Promedio: {np.mean(stomata_counts):.1f} ± {np.std(stomata_counts):.1f}")
    print(f"- Rango: {np.min(stomata_counts)} - {np.max(stomata_counts)}")
    
    print(f"\\nDensidad estomatal (est/mm²):")
    print(f"- Promedio: {np.mean(densities):.1f} ± {np.std(densities):.1f}")
    print(f"- Rango: {np.min(densities):.1f} - {np.max(densities):.1f}")
    
    print(f"\\nÁrea promedio (px²):")
    print(f"- Promedio: {np.mean(avg_areas):.1f} ± {np.std(avg_areas):.1f}")
    print(f"- Rango: {np.min(avg_areas):.1f} - {np.max(avg_areas):.1f}")
    
    # Interpretación biológica
    print(f"\\nInterpretación biológica:")
    avg_density = np.mean(densities)
    if avg_density < 100:
        print("- Densidad estomatal BAJA (posible adaptación a alta humedad)")
    elif avg_density < 200:
        print("- Densidad estomatal NORMAL (condiciones estándar)")
    else:
        print("- Densidad estomatal ALTA (posible estrés hídrico o adaptación)")

def main():
    """Ejecutar todos los ejemplos"""
    print("🌿 EJEMPLOS DE USO - RECONOCEDOR DE ESTOMAS")
    print("=" * 60)
    
    try:
        ejemplo_analisis_individual()
        ejemplo_procesamiento_lote() 
        ejemplo_tiempo_real()
        ejemplo_configuracion_avanzada()
        ejemplo_analisis_estadistico()
        
        print("\\n" + "=" * 60)
        print("✅ Todos los ejemplos ejecutados correctamente")
        print("\\nPara usar el sistema:")
        print("1. Ejecutar GUI: python main.py --mode gui")
        print("2. Análisis individual: python main.py --mode single --input imagen.jpg")
        print("3. Procesamiento lote: python main.py --mode batch --input directorio/")
        print("4. Tiempo real: python main.py --mode realtime")
        
    except Exception as e:
        print(f"\\nError ejecutando ejemplos: {e}")
        print("Verifica que todas las dependencias están instaladas:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
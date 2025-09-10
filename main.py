"""
Archivo principal para ejecutar el reconocedor de estomas
"""
import sys
import os
import argparse

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gui_app import StomataGUI
from batch_processor import BatchStomataProcessor
from realtime_detector import RealtimeStomataDetector
from stomata_analyzer import StomataAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Reconocedor de Estomas - Sistema Completo')
    parser.add_argument('--mode', choices=['gui', 'batch', 'realtime', 'single'],
                        default='gui', help='Modo de ejecución')
    parser.add_argument('--input', type=str, help='Imagen o directorio de entrada')
    parser.add_argument('--output', type=str, default='output', help='Directorio de salida')
    parser.add_argument('--yolo-model', type=str, help='Ruta al modelo YOLO')
    parser.add_argument('--unet-model', type=str, help='Ruta al modelo U-Net')
    parser.add_argument('--confidence', type=float, default=0.5, help='Umbral de confianza')
    parser.add_argument('--pixel-size', type=float, default=1.0, help='Tamaño del píxel en μm')
    parser.add_argument('--scale-factor', type=float, default=1.0, help='Factor de escala')

    args = parser.parse_args()

    print("🌿 Reconocedor de Estomas - Sistema Completo")
    print("=" * 50)

    if args.mode == 'gui':
        print("Iniciando interfaz gráfica...")
        app = StomataGUI()
        app.run()

    elif args.mode == 'single':
        if not args.input:
            print("Error: Especifica una imagen con --input")
            return

        print(f"Analizando imagen individual: {args.input}")
        analyzer = StomataAnalyzer(
            yolo_model_path=args.yolo_model,
            unet_model_path=args.unet_model
        )

        import cv2
        image = cv2.imread(args.input)
        if image is None:
            print("Error: No se pudo cargar la imagen")
            return

        analysis = analyzer.analyze_image(
            image,
            pixel_size_um=args.pixel_size,
            scale_factor=args.scale_factor,
            confidence_threshold=args.confidence
        )

        print(f"Resultados del análisis:")
        print(f"- Total estomas: {analysis.total_count}")
        print(f"- Densidad estomatal: {analysis.stomatal_density:.2f} est/mm²")
        print(f"- Área promedio: {analysis.avg_area:.1f} px²")

        # Guardar visualización
        visualization = analyzer.visualize_analysis(image, analysis)
        output_path = os.path.join(args.output, "analysis_result.jpg")
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(output_path, visualization)
        print(f"Visualización guardada en: {output_path}")

    elif args.mode == 'batch':
        if not args.input:
            print("Error: Especifica un directorio con --input")
            return

        print(f"Procesando lote desde: {args.input}")
        processor = BatchStomataProcessor(
            yolo_model_path=args.yolo_model,
            unet_model_path=args.unet_model,
            output_dir=args.output,
            pixel_size_um=args.pixel_size
        )

        results = processor.process_image_batch(
            args.input,
            scale_factor=args.scale_factor,
            confidence_threshold=args.confidence
        )

        print(f"Procesamiento completado:")
        print(f"- Imágenes procesadas: {results['processed']}")
        print(f"- Imágenes fallidas: {results['failed']}")

        # Exportar resultados
        if results['results']:
            processor.export_to_csv(results['results'])
            processor.export_to_excel(results['results'], results['consolidated_report'])
            processor.generate_summary_report(results['results'], results['consolidated_report'])
            print(f"Reportes guardados en: {args.output}/reports/")

    elif args.mode == 'realtime':
        print("Iniciando detección en tiempo real...")
        detector = RealtimeStomataDetector(
            yolo_model_path=args.yolo_model,
            unet_model_path=args.unet_model,
            pixel_size_um=args.pixel_size,
            scale_factor=args.scale_factor
        )

        try:
            detector.run_window_display()
        except KeyboardInterrupt:
            print("\nDeteniendo detector...")
        finally:
            detector.cleanup()


if __name__ == "__main__":
    main()
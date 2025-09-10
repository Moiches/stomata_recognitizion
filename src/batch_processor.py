"""
Procesador en lote para analizar múltiples imágenes de estomas
"""
import os
import cv2
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np

from stomata_analyzer import StomataAnalyzer, StomataAnalysis


class BatchStomataProcessor:
    def _init_(self,
                 yolo_model_path: str = None,
                 unet_model_path: str = None,
                 output_dir: str = "output",
                 pixel_size_um: float = 1.0):
        """
        Inicializa el procesador en lote
        Args:
            yolo_model_path: Ruta al modelo YOLO entrenado
            unet_model_path: Ruta al modelo U-Net entrenado
            output_dir: Directorio de salida
            pixel_size_um: Tamaño del píxel en micrómetros
        """
        self.analyzer = StomataAnalyzer(
            yolo_model_path=yolo_model_path,
            unet_model_path=unet_model_path,
            use_yolo=True,
            use_unet=True
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.pixel_size_um = pixel_size_um

        # Crear subdirectorios
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

    def get_supported_image_files(self, directory: str) -> List[str]:
        """
        Obtiene lista de archivos de imagen soportados
        Args:
            directory: Directorio a escanear
        Returns:
            Lista de rutas de archivos
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    image_files.append(os.path.join(root, file))

        return sorted(image_files)

    def process_image_batch(self,
                            image_directory: str,
                            scale_factor: float = 1.0,
                            confidence_threshold: float = 0.5,
                            save_visualizations: bool = True,
                            save_individual_reports: bool = True) -> Dict:
        """
        Procesa un lote de imágenes
        Args:
            image_directory: Directorio con imágenes
            scale_factor: Factor de escala para calibración
            confidence_threshold: Umbral de confianza
            save_visualizations: Guardar imágenes con visualizaciones
            save_individual_reports: Guardar reportes individuales
        Returns:
            Diccionario con resultados del procesamiento
        """
        image_files = self.get_supported_image_files(image_directory)

        if not image_files:
            print(f"No se encontraron imágenes en {image_directory}")
            return {'processed': 0, 'results': []}

        print(f"Procesando {len(image_files)} imágenes...")

        batch_results = []
        failed_images = []

        for image_path in tqdm(image_files, desc="Analizando imágenes"):
            try:
                # Cargar imagen
                image = cv2.imread(image_path)
                if image is None:
                    failed_images.append(image_path)
                    continue

                # Analizar imagen
                analysis = self.analyzer.analyze_image(
                    image,
                    pixel_size_um=self.pixel_size_um,
                    scale_factor=scale_factor,
                    confidence_threshold=confidence_threshold
                )

                # Preparar resultado
                image_name = Path(image_path).stem
                result = {
                    'image_path': image_path,
                    'image_name': image_name,
                    'total_stomata': analysis.total_count,
                    'stomatal_density': analysis.stomatal_density,
                    'avg_area': analysis.avg_area,
                    'avg_circularity': analysis.avg_circularity,
                    'image_area_mm2': analysis.image_area_mm2,
                    'statistics': analysis.statistics
                }

                batch_results.append(result)

                # Guardar visualización
                if save_visualizations:
                    visualization = self.analyzer.visualize_analysis(
                        image, analysis, show_stats=True
                    )
                    viz_path = self.output_dir / "visualizations" / f"{image_name}_analysis.jpg"
                    cv2.imwrite(str(viz_path), visualization)

                # Guardar reporte individual
                if save_individual_reports:
                    report_path = self.output_dir / "analysis" / f"{image_name}_analysis.json"
                    self.analyzer.save_analysis(analysis, str(report_path))

            except Exception as e:
                print(f"Error procesando {image_path}: {e}")
                failed_images.append(image_path)

        # Crear reporte consolidado
        consolidated_report = self._create_consolidated_report(batch_results)

        return {
            'processed': len(batch_results),
            'failed': len(failed_images),
            'failed_images': failed_images,
            'results': batch_results,
            'consolidated_report': consolidated_report
        }

    def _create_consolidated_report(self, results: List[Dict]) -> Dict:
        """
        Crea un reporte consolidado de todos los análisis
        Args:
            results: Lista de resultados individuales
        Returns:
            Reporte consolidado
        """
        if not results:
            return {}

        # Extraer métricas
        stomata_counts = [r['total_stomata'] for r in results]
        densities = [r['stomatal_density'] for r in results]
        avg_areas = [r['avg_area'] for r in results]
        circularities = [r['avg_circularity'] for r in results]

        # Calcular estadísticas generales
        report = {
            'total_images': len(results),
            'total_stomata_detected': sum(stomata_counts),
            'statistics': {
                'stomata_count': {
                    'mean': np.mean(stomata_counts),
                    'std': np.std(stomata_counts),
                    'min': np.min(stomata_counts),
                    'max': np.max(stomata_counts),
                    'median': np.median(stomata_counts)
                },
                'stomatal_density': {
                    'mean': np.mean(densities),
                    'std': np.std(densities),
                    'min': np.min(densities),
                    'max': np.max(densities),
                    'median': np.median(densities)
                },
                'average_area': {
                    'mean': np.mean(avg_areas),
                    'std': np.std(avg_areas),
                    'min': np.min(avg_areas),
                    'max': np.max(avg_areas),
                    'median': np.median(avg_areas)
                },
                'circularity': {
                    'mean': np.mean([c for c in circularities if c > 0]),
                    'std': np.std([c for c in circularities if c > 0]),
                    'min': np.min([c for c in circularities if c > 0]),
                    'max': np.max([c for c in circularities if c > 0])
                }
            }
        }

        return report

    def export_to_csv(self, results: List[Dict], filename: str = "stomata_analysis.csv"):
        """
        Exporta resultados a CSV
        Args:
            results: Lista de resultados
            filename: Nombre del archivo CSV
        """
        if not results:
            print("No hay resultados para exportar")
            return

        # Preparar datos para DataFrame
        df_data = []
        for result in results:
            row = {
                'image_name': result['image_name'],
                'image_path': result['image_path'],
                'total_stomata': result['total_stomata'],
                'stomatal_density_mm2': result['stomatal_density'],
                'avg_stomata_area_px2': result['avg_area'],
                'avg_circularity': result['avg_circularity'],
                'image_area_mm2': result['image_area_mm2']
            }

            # Añadir estadísticas si están disponibles
            if 'statistics' in result:
                stats = result['statistics']
                if 'area_stats' in stats:
                    row.update({
                        'area_mean': stats['area_stats']['mean'],
                        'area_std': stats['area_stats']['std'],
                        'area_min': stats['area_stats']['min'],
                        'area_max': stats['area_stats']['max']
                    })

                if 'size_distribution' in stats:
                    row.update({
                        'small_stomata': stats['size_distribution']['small'],
                        'medium_stomata': stats['size_distribution']['medium'],
                        'large_stomata': stats['size_distribution']['large']
                    })

            df_data.append(row)

        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / "reports" / filename
        df.to_csv(csv_path, index=False)
        print(f"Resultados exportados a {csv_path}")

    def export_to_excel(self,
                        results: List[Dict],
                        consolidated_report: Dict,
                        filename: str = "stomata_analysis.xlsx"):
        """
        Exporta resultados a Excel con múltiples hojas
        Args:
            results: Lista de resultados individuales
            consolidated_report: Reporte consolidado
            filename: Nombre del archivo Excel
        """
        if not results:
            print("No hay resultados para exportar")
            return

        excel_path = self.output_dir / "reports" / filename

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Hoja de resultados individuales
            df_results = pd.DataFrame([
                {
                    'Imagen': r['image_name'],
                    'Ruta': r['image_path'],
                    'Total_Estomas': r['total_stomata'],
                    'Densidad_Estomatal_mm2': r['stomatal_density'],
                    'Área_Promedio_px2': r['avg_area'],
                    'Circularidad_Promedio': r['avg_circularity'],
                    'Área_Imagen_mm2': r['image_area_mm2']
                } for r in results
            ])
            df_results.to_excel(writer, sheet_name='Resultados_Individuales', index=False)

            # Hoja de estadísticas consolidadas
            if consolidated_report and 'statistics' in consolidated_report:
                stats_data = []
                for metric, values in consolidated_report['statistics'].items():
                    stats_data.append({
                        'Métrica': metric.replace('_', ' ').title(),
                        'Media': values.get('mean', 0),
                        'Desv_Estándar': values.get('std', 0),
                        'Mínimo': values.get('min', 0),
                        'Máximo': values.get('max', 0),
                        'Mediana': values.get('median', 0)
                    })

                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Estadísticas_Consolidadas', index=False)

        print(f"Reporte completo exportado a {excel_path}")

    def generate_summary_report(self,
                                results: List[Dict],
                                consolidated_report: Dict,
                                filename: str = "summary_report.txt"):
        """
        Genera un reporte resumen en texto
        Args:
            results: Lista de resultados
            consolidated_report: Reporte consolidado
            filename: Nombre del archivo de reporte
        """
        report_path = self.output_dir / "reports" / filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ANÁLISIS DE ESTOMAS\n")
            f.write("=" * 50 + "\n\n")

            if consolidated_report:
                f.write(f"Total de imágenes analizadas: {consolidated_report['total_images']}\n")
                f.write(f"Total de estomas detectados: {consolidated_report['total_stomata_detected']}\n\n")

                f.write("ESTADÍSTICAS GENERALES:\n")
                f.write("-" * 25 + "\n")

                stats = consolidated_report['statistics']

                if 'stomata_count' in stats:
                    sc = stats['stomata_count']
                    f.write(f"Conteo de estomas por imagen:\n")
                    f.write(f"  - Promedio: {sc['mean']:.2f} ± {sc['std']:.2f}\n")
                    f.write(f"  - Rango: {sc['min']:.0f} - {sc['max']:.0f}\n")
                    f.write(f"  - Mediana: {sc['median']:.2f}\n\n")

                if 'stomatal_density' in stats:
                    sd = stats['stomatal_density']
                    f.write(f"Densidad estomatal (estomas/mm²):\n")
                    f.write(f"  - Promedio: {sd['mean']:.2f} ± {sd['std']:.2f}\n")
                    f.write(f"  - Rango: {sd['min']:.2f} - {sd['max']:.2f}\n")
                    f.write(f"  - Mediana: {sd['median']:.2f}\n\n")

            f.write("RESULTADOS INDIVIDUALES:\n")
            f.write("-" * 25 + "\n")
            for result in results:
                f.write(f"Imagen: {result['image_name']}\n")
                f.write(f"  - Estomas: {result['total_stomata']}\n")
                f.write(f"  - Densidad: {result['stomatal_density']:.2f} est/mm²\n")
                f.write(f"  - Área promedio: {result['avg_area']:.1f} px²\n")
                f.write("\n")

        print(f"Reporte resumen generado en {report_path}")
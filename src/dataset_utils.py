"""
Utilidades para preparación y manejo de datasets de estomas
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
import shutil
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class StomataDatasetManager:
    def __init__(self, base_dir: str):
        """
        Inicializa el gestor de datasets
        Args:
            base_dir: Directorio base para datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

    def create_dataset_structure(self, dataset_name: str) -> Path:
        """
        Crea la estructura de directorios para un dataset
        Args:
            dataset_name: Nombre del dataset
        Returns:
            Ruta al directorio del dataset
        """
        dataset_path = self.base_dir / dataset_name

        # Crear estructura estándar
        dirs_to_create = [
            'raw_images',  # Imágenes originales sin anotar
            'annotated_images',  # Imágenes con anotaciones
            'labels',  # Archivos .txt con anotaciones YOLO
            'augmented',  # Imágenes aumentadas
            'splits',  # Divisiones train/val/test
            'metadata'  # Información adicional
        ]

        for dir_name in dirs_to_create:
            (dataset_path / dir_name).mkdir(exist_ok=True, parents=True)

        # Crear archivo de configuración del dataset
        config = {
            'dataset_name': dataset_name,
            'created_date': pd.Timestamp.now().isoformat(),
            'description': f'Dataset de estomas: {dataset_name}',
            'classes': ['stomata'],
            'num_classes': 1,
            'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            'annotation_format': 'YOLO',
            'statistics': {
                'total_images': 0,
                'annotated_images': 0,
                'total_annotations': 0,
                'avg_annotations_per_image': 0.0
            }
        }

        config_path = dataset_path / 'dataset_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"📁 Estructura de dataset creada en: {dataset_path}")
        return dataset_path

    def import_images(self, dataset_path: Path, source_dir: str) -> Dict:
        """
        Importa imágenes desde un directorio fuente
        Args:
            dataset_path: Ruta del dataset
            source_dir: Directorio fuente con imágenes
        Returns:
            Estadísticas de importación
        """
        source_path = Path(source_dir)
        target_path = dataset_path / 'raw_images'

        # Extensiones soportadas
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # Buscar imágenes recursivamente
        image_files = []
        for ext in extensions:
            image_files.extend(source_path.rglob(f'*{ext}'))
            image_files.extend(source_path.rglob(f'*{ext.upper()}'))

        print(f"📥 Importando {len(image_files)} imágenes...")

        imported = 0
        skipped = 0

        for img_path in tqdm(image_files):
            try:
                # Verificar que la imagen se pueda cargar
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue

                # Copiar imagen con nombre único
                new_name = f"{img_path.stem}_{img_path.suffix}"
                counter = 1
                while (target_path / new_name).exists():
                    new_name = f"{img_path.stem}_{counter}{img_path.suffix}"
                    counter += 1

                target_file = target_path / new_name
                shutil.copy2(img_path, target_file)
                imported += 1

            except Exception as e:
                print(f"⚠️ Error importando {img_path}: {e}")
                skipped += 1

        stats = {
            'imported': imported,
            'skipped': skipped,
            'total_found': len(image_files)
        }

        print(f"✅ Importación completada: {imported} imágenes importadas, {skipped} omitidas")
        return stats

    def validate_annotations(self, dataset_path: Path) -> Dict:
        """
        Valida las anotaciones del dataset
        Args:
            dataset_path: Ruta del dataset
        Returns:
            Reporte de validación
        """
        print("🔍 Validando anotaciones...")

        labels_dir = dataset_path / 'labels'
        images_dir = dataset_path / 'annotated_images'

        validation_results = {
            'valid_annotations': 0,
            'invalid_annotations': 0,
            'missing_images': 0,
            'missing_labels': 0,
            'empty_labels': 0,
            'annotation_stats': {
                'total_boxes': 0,
                'boxes_per_image': [],
                'box_sizes': [],
                'center_positions': []
            },
            'issues': []
        }

        # Obtener archivos de anotaciones
        label_files = list(labels_dir.glob('*.txt'))

        for label_file in tqdm(label_files):
            try:
                # Verificar que existe imagen correspondiente
                img_name = label_file.stem
                img_files = [
                    images_dir / f"{img_name}.jpg",
                    images_dir / f"{img_name}.jpeg",
                    images_dir / f"{img_name}.png"
                ]

                img_path = None
                for img_file in img_files:
                    if img_file.exists():
                        img_path = img_file
                        break

                if img_path is None:
                    validation_results['missing_images'] += 1
                    validation_results['issues'].append(f"Imagen faltante para: {label_file}")
                    continue

                # Cargar imagen para obtener dimensiones
                img = cv2.imread(str(img_path))
                if img is None:
                    validation_results['issues'].append(f"No se puede cargar imagen: {img_path}")
                    continue

                h, w = img.shape[:2]

                # Leer anotaciones
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    validation_results['empty_labels'] += 1
                    continue

                boxes_in_image = 0
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        validation_results['issues'].append(
                            f"Formato incorrecto en {label_file}: {line.strip()}"
                        )
                        continue

                    try:
                        class_id, center_x, center_y, width, height = map(float, parts)

                        # Validar rangos
                        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                                0 <= width <= 1 and 0 <= height <= 1):
                            validation_results['issues'].append(
                                f"Coordenadas fuera de rango en {label_file}: {line.strip()}"
                            )
                            continue

                        boxes_in_image += 1
                        validation_results['annotation_stats']['total_boxes'] += 1
                        validation_results['annotation_stats']['box_sizes'].append((width, height))
                        validation_results['annotation_stats']['center_positions'].append((center_x, center_y))

                    except ValueError:
                        validation_results['issues'].append(
                            f"Valores no numéricos en {label_file}: {line.strip()}"
                        )
                        continue

                validation_results['annotation_stats']['boxes_per_image'].append(boxes_in_image)
                validation_results['valid_annotations'] += 1

            except Exception as e:
                validation_results['invalid_annotations'] += 1
                validation_results['issues'].append(f"Error procesando {label_file}: {e}")

        # Calcular estadísticas
        if validation_results['annotation_stats']['boxes_per_image']:
            validation_results['annotation_stats']['avg_boxes_per_image'] = np.mean(
                validation_results['annotation_stats']['boxes_per_image']
            )

        print(f"📊 Validación completada:")
        print(f"  - Anotaciones válidas: {validation_results['valid_annotations']}")
        print(f"  - Anotaciones inválidas: {validation_results['invalid_annotations']}")
        print(f"  - Total de cajas: {validation_results['annotation_stats']['total_boxes']}")
        print(f"  - Problemas encontrados: {len(validation_results['issues'])}")

        return validation_results

    def generate_dataset_statistics(self, dataset_path: Path) -> None:
        """
        Genera estadísticas y visualizaciones del dataset
        Args:
            dataset_path: Ruta del dataset
        """
        print("📊 Generando estadísticas del dataset...")

        validation_results = self.validate_annotations(dataset_path)
        stats_dir = dataset_path / 'statistics'
        stats_dir.mkdir(exist_ok=True)

        # Crear visualizaciones
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Estadísticas del Dataset de Estomas', fontsize=16)

        # 1. Distribución de número de estomas por imagen
        if validation_results['annotation_stats']['boxes_per_image']:
            axes[0, 0].hist(validation_results['annotation_stats']['boxes_per_image'],
                            bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title('Distribución de Estomas por Imagen')
            axes[0, 0].set_xlabel('Número de Estomas')
            axes[0, 0].set_ylabel('Frecuencia')

        # 2. Distribución de tamaños de cajas
        if validation_results['annotation_stats']['box_sizes']:
            widths, heights = zip(*validation_results['annotation_stats']['box_sizes'])
            areas = [w * h for w, h in validation_results['annotation_stats']['box_sizes']]

            axes[0, 1].scatter(widths, heights, alpha=0.6)
            axes[0, 1].set_title('Tamaños de Anotaciones')
            axes[0, 1].set_xlabel('Ancho (normalizado)')
            axes[0, 1].set_ylabel('Alto (normalizado)')

        # 3. Distribución de posiciones
        if validation_results['annotation_stats']['center_positions']:
            x_centers, y_centers = zip(*validation_results['annotation_stats']['center_positions'])

            axes[1, 0].scatter(x_centers, y_centers, alpha=0.6)
            axes[1, 0].set_title('Distribución Espacial de Estomas')
            axes[1, 0].set_xlabel('Posición X (normalizada)')
            axes[1, 0].set_ylabel('Posición Y (normalizada)')
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)

        # 4. Resumen estadístico
        summary_text = f"""
        Dataset: {dataset_path.name}

        Imágenes anotadas: {validation_results['valid_annotations']}
        Total de estomas: {validation_results['annotation_stats']['total_boxes']}

        Promedio estomas/imagen: {validation_results['annotation_stats'].get('avg_boxes_per_image', 0):.2f}

        Problemas encontrados: {len(validation_results['issues'])}
        """

        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(stats_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Guardar estadísticas en JSON
        stats_file = stats_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            # Convertir arrays numpy a listas para JSON
            stats_copy = validation_results.copy()
            stats_copy['annotation_stats']['boxes_per_image'] = \
                validation_results['annotation_stats']['boxes_per_image']

            json.dump(stats_copy, f, indent=2, default=str)

        print(f"📊 Estadísticas guardadas en: {stats_dir}")

    def augment_dataset(self, dataset_path: Path,
                        augmentation_factor: int = 3) -> None:
        """
        Aplica aumento de datos al dataset
        Args:
            dataset_path: Ruta del dataset
            augmentation_factor: Factor de multiplicación de datos
        """
        print(f"🔄 Aplicando aumento de datos (factor: {augmentation_factor})...")

        images_dir = dataset_path / 'annotated_images'
        labels_dir = dataset_path / 'labels'
        aug_images_dir = dataset_path / 'augmented' / 'images'
        aug_labels_dir = dataset_path / 'augmented' / 'labels'

        aug_images_dir.mkdir(exist_ok=True, parents=True)
        aug_labels_dir.mkdir(exist_ok=True, parents=True)

        # Obtener pares imagen-etiqueta
        image_files = list(images_dir.glob('*.jpg')) + \
                      list(images_dir.glob('*.png')) + \
                      list(images_dir.glob('*.jpeg'))

        augmentations = [
            self._flip_horizontal,
            self._flip_vertical,
            self._rotate_90,
            self._adjust_brightness,
            self._adjust_contrast,
            self._add_noise
        ]

        for img_path in tqdm(image_files):
            label_path = labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            # Cargar imagen y anotaciones
            image = cv2.imread(str(img_path))
            with open(label_path, 'r') as f:
                annotations = f.readlines()

            # Aplicar aumentos
            for i in range(augmentation_factor):
                aug_func = np.random.choice(augmentations)
                aug_image, aug_annotations = aug_func(image, annotations)

                # Guardar imagen aumentada
                aug_img_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                aug_img_path = aug_images_dir / aug_img_name
                cv2.imwrite(str(aug_img_path), aug_image)

                # Guardar anotaciones aumentadas
                aug_label_path = aug_labels_dir / f"{img_path.stem}_aug_{i}.txt"
                with open(aug_label_path, 'w') as f:
                    f.writelines(aug_annotations)

        print(f"✅ Aumento de datos completado")

    def _flip_horizontal(self, image, annotations):
        """Volteo horizontal"""
        flipped_img = cv2.flip(image, 1)
        flipped_annotations = []

        for line in annotations:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, center_x, center_y, width, height = parts
                # Ajustar coordenada X
                new_center_x = 1.0 - float(center_x)
                flipped_annotations.append(
                    f"{class_id} {new_center_x:.6f} {center_y} {width} {height}\n"
                )

        return flipped_img, flipped_annotations

    def _flip_vertical(self, image, annotations):
        """Volteo vertical"""
        flipped_img = cv2.flip(image, 0)
        flipped_annotations = []

        for line in annotations:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, center_x, center_y, width, height = parts
                # Ajustar coordenada Y
                new_center_y = 1.0 - float(center_y)
                flipped_annotations.append(
                    f"{class_id} {center_x} {new_center_y:.6f} {width} {height}\n"
                )

        return flipped_img, flipped_annotations

    def _rotate_90(self, image, annotations):
        """Rotación 90 grados"""
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_annotations = []

        for line in annotations:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, center_x, center_y, width, height = parts
                # Rotar coordenadas
                new_center_x = 1.0 - float(center_y)
                new_center_y = float(center_x)
                new_width = height
                new_height = width

                rotated_annotations.append(
                    f"{class_id} {new_center_x:.6f} {new_center_y:.6f} {new_width} {new_height}\n"
                )

        return rotated_img, rotated_annotations

    def _adjust_brightness(self, image, annotations):
        """Ajuste de brillo"""
        factor = np.random.uniform(0.7, 1.3)
        brightened = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return brightened, annotations

    def _adjust_contrast(self, image, annotations):
        """Ajuste de contraste"""
        factor = np.random.uniform(0.8, 1.2)
        contrasted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return contrasted, annotations

    def _add_noise(self, image, annotations):
        """Agregar ruido"""
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy, annotations


if __name__ == "__main__":
    # Ejemplo de uso
    manager = StomataDatasetManager("datasets")

    # Crear dataset
    dataset_path = manager.create_dataset_structure("my_stomata_dataset")

    # Importar imágenes
    # manager.import_images(dataset_path, "path/to/your/images")

    # Validar anotaciones
    # manager.validate_annotations(dataset_path)

    # Generar estadísticas
    # manager.generate_dataset_statistics(dataset_path)
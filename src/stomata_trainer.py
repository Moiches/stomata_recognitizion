"""
Entrenador especializado para modelos de reconocimiento de estomas
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


class StomataYOLOTrainer:
    def __init__(self,
                 dataset_path: str,
                 output_dir: str = "trained_models",
                 base_model: str = "yolov8n.pt"):
        """
        Inicializa el entrenador de YOLO para estomas
        Args:
            dataset_path: Ruta al dataset anotado
            output_dir: Directorio para guardar modelos entrenados
            base_model: Modelo base de YOLO (yolov8n, yolov8s, yolov8m, yolov8l)
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.base_model = base_model

        # Configuración por defecto
        self.config = {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'learning_rate': 0.001,
            'patience': 50,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    def prepare_yolo_dataset(self, train_split: float = 0.8, val_split: float = 0.1) -> str:
        """
        Prepara el dataset en formato YOLO
        Args:
            train_split: Proporción para entrenamiento
            val_split: Proporción para validación (resto será test)
        Returns:
            Ruta al archivo data.yaml
        """
        print("📁 Preparando dataset en formato YOLO...")

        # Crear estructura de directorios
        dataset_dir = self.output_dir / "dataset"
        for split in ['train', 'val', 'test']:
            (dataset_dir / split / 'images').mkdir(exist_ok=True, parents=True)
            (dataset_dir / split / 'labels').mkdir(exist_ok=True, parents=True)

        # Obtener lista de imágenes anotadas
        image_files = list(self.dataset_path.glob("**/*.jpg")) + \
                      list(self.dataset_path.glob("**/*.png")) + \
                      list(self.dataset_path.glob("**/*.jpeg"))

        # Filtrar solo imágenes que tienen anotaciones
        annotated_images = []
        for img_path in image_files:
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                annotated_images.append(img_path)

        if not annotated_images:
            raise ValueError(f"No se encontraron imágenes anotadas en {self.dataset_path}")

        print(f"📊 Encontradas {len(annotated_images)} imágenes anotadas")

        # Dividir dataset
        np.random.seed(42)
        np.random.shuffle(annotated_images)

        n_train = int(len(annotated_images) * train_split)
        n_val = int(len(annotated_images) * val_split)

        splits = {
            'train': annotated_images[:n_train],
            'val': annotated_images[n_train:n_train + n_val],
            'test': annotated_images[n_train + n_val:]
        }

        # Copiar archivos a directorios correspondientes
        for split_name, img_list in splits.items():
            print(f"Procesando {split_name}: {len(img_list)} imágenes")
            for img_path in tqdm(img_list):
                # Copiar imagen
                dst_img = dataset_dir / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dst_img)

                # Copiar anotación
                label_path = img_path.with_suffix('.txt')
                dst_label = dataset_dir / split_name / 'labels' / label_path.name
                if label_path.exists():
                    shutil.copy2(label_path, dst_label)

        # Crear archivo data.yaml
        data_yaml = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # Número de clases (solo estomas)
            'names': ['stomata']
        }

        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"✅ Dataset preparado en: {dataset_dir}")
        print(f"📝 Configuración guardada en: {yaml_path}")

        return str(yaml_path)

    def train_model(self,
                    data_yaml: str,
                    epochs: int = None,
                    batch_size: int = None,
                    img_size: int = None,
                    resume: bool = False) -> str:
        """
        Entrena el modelo YOLO para estomas
        Args:
            data_yaml: Ruta al archivo de configuración del dataset
            epochs: Número de épocas (None usa config)
            batch_size: Tamaño de lote (None usa config)
            img_size: Tamaño de imagen (None usa config)
            resume: Continuar entrenamiento previo
        Returns:
            Ruta al modelo entrenado
        """
        # Usar configuración personalizada o por defecto
        epochs = epochs or self.config['epochs']
        batch_size = batch_size or self.config['batch_size']
        img_size = img_size or self.config['img_size']

        print(f"🚀 Iniciando entrenamiento de YOLO para estomas...")
        print(f"📊 Configuración:")
        print(f"  - Modelo base: {self.base_model}")
        print(f"  - Épocas: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Tamaño imagen: {img_size}")
        print(f"  - Device: {self.config['device']}")

        # Cargar modelo
        model = YOLO(self.base_model)

        # Entrenar
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.config['device'],
            project=str(self.output_dir),
            name='stomata_yolo',
            patience=self.config['patience'],
            save_period=10,
            resume=resume,
            plots=True,
            val=True,
            verbose=True
        )

        # Ruta del modelo entrenado
        model_path = self.output_dir / 'stomata_yolo' / 'weights' / 'best.pt'

        print(f"✅ Entrenamiento completado!")
        print(f"🎯 Modelo guardado en: {model_path}")

        return str(model_path)

    def evaluate_model(self, model_path: str, data_yaml: str) -> Dict:
        """
        Evalúa el modelo entrenado
        Args:
            model_path: Ruta al modelo entrenado
            data_yaml: Ruta al archivo de configuración del dataset
        Returns:
            Diccionario con métricas de evaluación
        """
        print("📊 Evaluando modelo entrenado...")

        model = YOLO(model_path)

        # Validación en conjunto de test
        results = model.val(
            data=data_yaml,
            split='test',
            save_json=True,
            plots=True
        )

        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'fitness': results.fitness
        }

        print("📈 Resultados de evaluación:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")

        return metrics

    def test_on_sample_images(self,
                              model_path: str,
                              test_images_dir: str,
                              confidence: float = 0.5) -> None:
        """
        Prueba el modelo en imágenes de muestra
        Args:
            model_path: Ruta al modelo entrenado
            test_images_dir: Directorio con imágenes de prueba
            confidence: Umbral de confianza
        """
        print(f"🔍 Probando modelo en imágenes de muestra...")

        model = YOLO(model_path)
        test_dir = Path(test_images_dir)
        output_dir = self.output_dir / 'sample_predictions'
        output_dir.mkdir(exist_ok=True)

        # Obtener imágenes de prueba
        image_files = list(test_dir.glob("*.jpg")) + \
                      list(test_dir.glob("*.png")) + \
                      list(test_dir.glob("*.jpeg"))

        for img_path in tqdm(image_files[:10]):  # Limitar a 10 imágenes
            # Predicción
            results = model(str(img_path), conf=confidence)

            # Guardar imagen con detecciones
            for i, result in enumerate(results):
                output_path = output_dir / f"{img_path.stem}_prediction.jpg"
                result.save(str(output_path))

                # Mostrar estadísticas
                boxes = result.boxes
                if boxes is not None:
                    n_detections = len(boxes)
                    confidences = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
                    print(f"  {img_path.name}: {n_detections} estomas detectados, "
                          f"confianza promedio: {np.mean(confidences):.3f}")
                else:
                    print(f"  {img_path.name}: 0 estomas detectados")

        print(f"💾 Resultados guardados en: {output_dir}")

    def create_training_report(self, model_path: str, metrics: Dict) -> str:
        """
        Crea un reporte del entrenamiento
        Args:
            model_path: Ruta al modelo entrenado
            metrics: Métricas de evaluación
        Returns:
            Ruta al reporte generado
        """
        report_path = self.output_dir / 'training_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Reporte de Entrenamiento - Modelo YOLO para Estomas\n\n")

            f.write("## Configuración del Entrenamiento\n")
            f.write(f"- **Modelo base**: {self.base_model}\n")
            f.write(f"- **Épocas**: {self.config['epochs']}\n")
            f.write(f"- **Batch size**: {self.config['batch_size']}\n")
            f.write(f"- **Tamaño imagen**: {self.config['img_size']}\n")
            f.write(f"- **Device**: {self.config['device']}\n\n")

            f.write("## Métricas de Evaluación\n")
            for metric, value in metrics.items():
                f.write(f"- **{metric}**: {value:.4f}\n")
            f.write("\n")

            f.write("## Uso del Modelo Entrenado\n")
            f.write("```python\n")
            f.write("from src.yolo_detector import StomataYOLODetector\n")
            f.write(f"detector = StomataYOLODetector('{model_path}')\n")
            f.write("detections = detector.detect_stomata(image)\n")
            f.write("```\n\n")

            f.write("## Integración con el Sistema\n")
            f.write("```bash\n")
            f.write(f"python main.py --mode single --yolo-model {model_path} --input imagen.jpg\n")
            f.write("```\n")

        print(f"📄 Reporte generado en: {report_path}")
        return str(report_path)


def create_sample_annotation_tool():
    """
    Crea una herramienta simple para anotar imágenes de estomas
    """
    annotation_code = '''
"""
Herramienta simple para anotar estomas en imágenes
Uso: python annotate_stomata.py <imagen> <output_dir>
"""
import cv2
import numpy as np
import sys
from pathlib import Path

class StomataAnnotator:
    def __init__(self, image_path, output_dir):
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.image = cv2.imread(image_path)
        self.annotations = []
        self.current_box = None
        self.drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            temp_img = self.image.copy()
            cv2.rectangle(temp_img, self.current_box[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('Anotar Estomas', temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box.append((x, y))
            self.annotations.append(self.current_box)
            self.draw_annotations()

    def draw_annotations(self):
        temp_img = self.image.copy()
        for box in self.annotations:
            cv2.rectangle(temp_img, box[0], box[1], (0, 255, 0), 2)
        cv2.imshow('Anotar Estomas', temp_img)

    def save_annotations(self):
        if not self.annotations:
            return

        # Formato YOLO: class_id center_x center_y width height (normalizados)
        h, w = self.image.shape[:2]

        label_path = self.output_dir / f"{Path(self.image_path).stem}.txt"
        with open(label_path, 'w') as f:
            for box in self.annotations:
                x1, y1 = box[0]
                x2, y2 = box[1]

                # Convertir a formato YOLO
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                width = abs(x2 - x1) / w
                height = abs(y2 - y1) / h

                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\\n")

        print(f"Anotaciones guardadas: {label_path}")

    def annotate(self):
        cv2.namedWindow('Anotar Estomas', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Anotar Estomas', self.mouse_callback)
        cv2.imshow('Anotar Estomas', self.image)

        print("Instrucciones:")
        print("- Arrastra para crear cajas alrededor de estomas")
        print("- Presiona 's' para guardar")
        print("- Presiona 'r' para reiniciar")
        print("- Presiona 'q' para salir")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_annotations()
                print(f"Guardadas {len(self.annotations)} anotaciones")
            elif key == ord('r'):
                self.annotations = []
                cv2.imshow('Anotar Estomas', self.image)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python annotate_stomata.py <imagen> <output_dir>")
        sys.exit(1)

    annotator = StomataAnnotator(sys.argv[1], sys.argv[2])
    annotator.annotate()
'''

    with open("annotate_stomata.py", "w") as f:
        f.write(annotation_code)

    print("📝 Herramienta de anotación creada: annotate_stomata.py")


if __name__ == "__main__":
    # Ejemplo de uso
    trainer = StomataYOLOTrainer(
        dataset_path="data/stomata_dataset",
        output_dir="models/trained"
    )

    # Preparar dataset
    data_yaml = trainer.prepare_yolo_dataset()

    # Entrenar modelo
    model_path = trainer.train_model(data_yaml, epochs=50)

    # Evaluar modelo
    metrics = trainer.evaluate_model(model_path, data_yaml)

    # Crear reporte
    trainer.create_training_report(model_path, metrics)
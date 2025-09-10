"""
Script principal para entrenar modelos YOLO específicos para estomas
"""
import argparse
import sys
import os
from pathlib import Path
import json

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stomata_trainer import StomataYOLOTrainer, create_sample_annotation_tool
from dataset_utils import StomataDatasetManager


def main():
    parser = argparse.ArgumentParser(description='Entrenador de YOLO para Estomas')
    parser.add_argument('--action', choices=['create-dataset', 'annotate', 'train', 'evaluate', 'full-pipeline'],
                        required=True, help='Acción a realizar')

    # Argumentos para dataset
    parser.add_argument('--dataset-name', type=str, default='stomata_dataset',
                        help='Nombre del dataset')
    parser.add_argument('--source-images', type=str,
                        help='Directorio con imágenes fuente para importar')
    parser.add_argument('--dataset-path', type=str,
                        help='Ruta al dataset existente')

    # Argumentos para entrenamiento
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n',
                        help='Tamaño del modelo YOLO (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Tamaño de lote')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Tamaño de imagen para entrenamiento')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                        help='Directorio de salida para modelos entrenados')

    # Argumentos para evaluación
    parser.add_argument('--model-path', type=str,
                        help='Ruta al modelo entrenado para evaluación')
    parser.add_argument('--test-images', type=str,
                        help='Directorio con imágenes de prueba')

    args = parser.parse_args()

    print("🌿 Entrenador de YOLO para Reconocimiento de Estomas")
    print("=" * 60)

    if args.action == 'create-dataset':
        create_dataset_action(args)
    elif args.action == 'annotate':
        create_annotation_tool_action(args)
    elif args.action == 'train':
        train_model_action(args)
    elif args.action == 'evaluate':
        evaluate_model_action(args)
    elif args.action == 'full-pipeline':
        full_pipeline_action(args)


def create_dataset_action(args):
    """Crea estructura de dataset y importa imágenes"""
    print(f"📁 Creando dataset: {args.dataset_name}")

    manager = StomataDatasetManager("datasets")
    dataset_path = manager.create_dataset_structure(args.dataset_name)

    if args.source_images:
        print(f"📥 Importando imágenes desde: {args.source_images}")
        stats = manager.import_images(dataset_path, args.source_images)
        print(f"✅ Importadas {stats['imported']} imágenes")

    print(f"\n📝 Siguiente paso: Anotar las imágenes")
    print(f"Comando sugerido:")
    print(f"python train_stomata_model.py --action annotate --dataset-path {dataset_path}")


def create_annotation_tool_action(args):
    """Crea herramienta de anotación"""
    print("📝 Creando herramienta de anotación...")
    create_sample_annotation_tool()

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        images_dir = dataset_path / 'raw_images'
        labels_dir = dataset_path / 'labels'
        annotated_dir = dataset_path / 'annotated_images'

        print(f"\n🎯 Para anotar tus imágenes:")
        print(f"1. Ejecuta: python annotate_stomata.py <imagen> {labels_dir}")
        print(f"2. Mueve imágenes anotadas a: {annotated_dir}")
        print(f"3. Las etiquetas se guardarán en: {labels_dir}")

        print(f"\n📊 Para validar anotaciones:")
        print(f"python train_stomata_model.py --action train --dataset-path {args.dataset_path}")


def train_model_action(args):
    """Entrena el modelo YOLO"""
    if not args.dataset_path:
        print("❌ Error: --dataset-path es requerido para entrenar")
        return

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset no encontrado en {dataset_path}")
        return

    print(f"🚀 Iniciando entrenamiento...")
    print(f"📊 Configuración:")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Modelo: YOLOv8{args.model_size}")
    print(f"  - Épocas: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Tamaño imagen: {args.img_size}")

    # Validar dataset primero
    print("\n📋 Validando dataset...")
    manager = StomataDatasetManager("datasets")
    validation_results = manager.validate_annotations(dataset_path)

    if validation_results['valid_annotations'] == 0:
        print("❌ Error: No se encontraron anotaciones válidas")
        print("Asegúrate de haber anotado las imágenes correctamente")
        return

    if len(validation_results['issues']) > 0:
        print(f"⚠️ Se encontraron {len(validation_results['issues'])} problemas:")
        for issue in validation_results['issues'][:5]:  # Mostrar primeros 5
            print(f"  - {issue}")
        if len(validation_results['issues']) > 5:
            print(f"  ... y {len(validation_results['issues']) - 5} más")

        response = input("\n¿Continuar con el entrenamiento? (y/n): ")
        if response.lower() != 'y':
            return

    # Generar estadísticas
    manager.generate_dataset_statistics(dataset_path)

    # Inicializar entrenador
    base_model = f"yolov8{args.model_size}.pt"
    trainer = StomataYOLOTrainer(
        dataset_path=str(dataset_path / 'annotated_images'),
        output_dir=args.output_dir,
        base_model=base_model
    )

    # Preparar dataset YOLO
    print("\n📦 Preparando dataset para YOLO...")
    data_yaml = trainer.prepare_yolo_dataset()

    # Entrenar modelo
    print("\n🏃 Iniciando entrenamiento...")
    model_path = trainer.train_model(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    # Evaluar modelo
    print("\n📊 Evaluando modelo...")
    metrics = trainer.evaluate_model(model_path, data_yaml)

    # Crear reporte
    report_path = trainer.create_training_report(model_path, metrics)

    print(f"\n✅ Entrenamiento completado!")
    print(f"🎯 Modelo guardado en: {model_path}")
    print(f"📄 Reporte en: {report_path}")

    print(f"\n🧪 Para probar el modelo:")
    print(f"python main.py --mode single --yolo-model {model_path} --input <imagen.jpg>")


def evaluate_model_action(args):
    """Evalúa un modelo existente"""
    if not args.model_path:
        print("❌ Error: --model-path es requerido para evaluación")
        return

    if not args.dataset_path:
        print("❌ Error: --dataset-path es requerido para evaluación")
        return

    model_path = Path(args.model_path)
    dataset_path = Path(args.dataset_path)

    if not model_path.exists():
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        return

    if not dataset_path.exists():
        print(f"❌ Error: Dataset no encontrado en {dataset_path}")
        return

    print(f"📊 Evaluando modelo: {model_path}")

    trainer = StomataYOLOTrainer(
        dataset_path=str(dataset_path),
        output_dir=args.output_dir
    )

    # Buscar archivo data.yaml
    data_yaml = dataset_path / 'data.yaml'
    if not data_yaml.exists():
        # Crear data.yaml si no existe
        data_yaml = trainer.prepare_yolo_dataset()

    # Evaluar
    metrics = trainer.evaluate_model(str(model_path), str(data_yaml))

    # Probar en imágenes de muestra si se proporciona
    if args.test_images:
        trainer.test_on_sample_images(str(model_path), args.test_images)

    print("✅ Evaluación completada")


def full_pipeline_action(args):
    """Ejecuta el pipeline completo"""
    if not args.source_images:
        print("❌ Error: --source-images es requerido para pipeline completo")
        return

    print("🔄 Ejecutando pipeline completo de entrenamiento...")

    # 1. Crear dataset
    print("\n📁 Paso 1: Creando dataset")
    args.action = 'create-dataset'
    create_dataset_action(args)

    # 2. Crear herramienta de anotación
    print("\n📝 Paso 2: Preparando herramientas de anotación")
    dataset_path = Path("datasets") / args.dataset_name
    args.dataset_path = str(dataset_path)
    create_annotation_tool_action(args)

    print(f"\n⏸️  PAUSA REQUERIDA")
    print(f"Antes de continuar, debes:")
    print(f"1. Anotar las imágenes usando la herramienta creada")
    print(f"2. Mover las imágenes anotadas al directorio correspondiente")
    print(f"")
    print(f"Cuando hayas terminado de anotar, ejecuta:")
    print(f"python train_stomata_model.py --action train --dataset-path {dataset_path}")


def create_config_template():
    """Crea un template de configuración"""
    config = {
        "training": {
            "model_size": "n",
            "epochs": 100,
            "batch_size": 16,
            "img_size": 640,
            "learning_rate": 0.001,
            "patience": 50
        },
        "dataset": {
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "augmentation_factor": 3,
            "min_annotations_per_image": 1
        },
        "paths": {
            "datasets_dir": "datasets",
            "output_dir": "trained_models",
            "base_models_dir": "models"
        }
    }

    config_path = "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"📄 Template de configuración creado: {config_path}")


if __name__ == "__main__":
    main()
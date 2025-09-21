"""
Script para mover imagenes anotadas desde raw_images a annotated_images
"""
import os
import shutil
from pathlib import Path

def move_annotated_images(dataset_path):
    """
    Mueve las imagenes que tienen labels desde raw_images a annotated_images
    """
    dataset_dir = Path(dataset_path)
    raw_images_dir = dataset_dir / 'raw_images'
    annotated_images_dir = dataset_dir / 'annotated_images'
    labels_dir = dataset_dir / 'labels'

    # Crear directorio annotated_images si no existe
    annotated_images_dir.mkdir(exist_ok=True, parents=True)

    print(f"Buscando labels en: {labels_dir}")
    print(f"Buscando imagenes en: {raw_images_dir}")
    print(f"Moviendo a: {annotated_images_dir}")
    print("-" * 50)

    # Obtener todos los archivos .txt (labels)
    label_files = list(labels_dir.glob('*.txt'))

    if not label_files:
        print("❌ No se encontraron archivos .txt en el directorio labels")
        return

    moved_count = 0
    not_found_count = 0

    for label_file in label_files:
        # Nombre base del archivo (sin extension)
        image_name = label_file.stem

        # Buscar imagen correspondiente con diferentes extensiones
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_found = False

        for ext in image_extensions:
            source_image = raw_images_dir / f"{image_name}{ext}"
            if source_image.exists():
                # Mover imagen a annotated_images
                dest_image = annotated_images_dir / f"{image_name}{ext}"

                try:
                    shutil.move(str(source_image), str(dest_image))
                    print(f"✅ Movido: {image_name}{ext}")
                    moved_count += 1
                    image_found = True
                    break
                except Exception as e:
                    print(f"❌ Error moviendo {image_name}{ext}: {e}")

        if not image_found:
            print(f"⚠️ No se encontro imagen para label: {image_name}.txt")
            not_found_count += 1

    print("-" * 50)
    print(f"📊 RESUMEN:")
    print(f"   Labels encontradas: {len(label_files)}")
    print(f"   Imagenes movidas: {moved_count}")
    print(f"   Imagenes no encontradas: {not_found_count}")
    print("-" * 50)

    if moved_count > 0:
        print(f"✅ {moved_count} imagenes movidas correctamente a annotated_images/")

    # Verificar contenido final
    final_images = list(annotated_images_dir.glob('*'))
    print(f"📁 Contenido final de annotated_images: {len(final_images)} archivos")

if __name__ == "__main__":
    dataset_path = "datasets/aglaia_cucullata"

    print("🔄 Moviendo imagenes anotadas...")
    print(f"Dataset: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"❌ Error: No se encuentra el dataset en {dataset_path}")
        exit(1)

    move_annotated_images(dataset_path)
    print("\n✅ Proceso completado!")
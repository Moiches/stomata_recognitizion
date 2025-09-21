"""
Herramienta simple para anotar estomas en imagenes
Uso: python annotate_stomata.py <imagen> <output_dir>
"""
import cv2
import numpy as np
import sys
import os
from pathlib import Path

class StomataAnnotator:
    def __init__(self, image_path, output_dir):
        # Convertir a ruta absoluta
        self.image_path = os.path.abspath(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Verificar que el archivo existe
        print(f"Intentando cargar imagen desde: {self.image_path}")
        if not os.path.exists(self.image_path):
            print(f"ERROR: El archivo no existe: {self.image_path}")
            print(f"Directorio actual: {os.getcwd()}")
            sys.exit(1)

        # Verificar que es un archivo de imagen
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        file_ext = os.path.splitext(self.image_path)[1].lower()
        if file_ext not in valid_extensions:
            print(f"ERROR: Extension no valida: {file_ext}")
            print(f"Extensiones soportadas: {valid_extensions}")
            sys.exit(1)

        # Cargar imagen
        self.image = cv2.imread(self.image_path)

        # Verificar que la imagen se cargo correctamente
        if self.image is None:
            print(f"ERROR: OpenCV no pudo cargar la imagen: {self.image_path}")
            print("Posibles causas:")
            print("- Archivo corrupto")
            print("- Formato no soportado por OpenCV")
            print("- Permisos de archivo")
            sys.exit(1)

        print(f"✅ Imagen cargada correctamente: {self.image.shape}")
        print(f"Tamaño: {self.image.shape[1]}x{self.image.shape[0]} pixeles")

        self.annotations = []
        self.current_box = None
        self.drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [(x, y)]
            print(f"Iniciando caja en: ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Mostrar todas las anotaciones existentes + la caja actual
            temp_img = self.image.copy()

            # Dibujar cajas existentes en verde
            for box in self.annotations:
                cv2.rectangle(temp_img, box[0], box[1], (0, 255, 0), 3)

            # Dibujar caja actual en rojo (temporal)
            cv2.rectangle(temp_img, self.current_box[0], (x, y), (0, 0, 255), 2)

            # Mostrar coordenadas en la imagen
            text = f"({x}, {y})"
            cv2.putText(temp_img, text, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Anotar Estomas', temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_x, end_y = x, y
            self.current_box.append((end_x, end_y))

            # Verificar que la caja tiene tamaño mínimo
            start_x, start_y = self.current_box[0]
            width = abs(end_x - start_x)
            height = abs(end_y - start_y)

            if width > 5 and height > 5:
                self.annotations.append(self.current_box)
                print(f"Caja añadida: ({start_x},{start_y}) -> ({end_x},{end_y}) | Tamaño: {width}x{height}")
                print(f"Total anotaciones: {len(self.annotations)}")
                self.draw_annotations()
            else:
                print(f"Caja muy pequeña ignorada: {width}x{height}")
                self.draw_annotations()

    def draw_annotations(self):
        temp_img = self.image.copy()

        # Dibujar cada caja con número
        for i, box in enumerate(self.annotations):
            # Caja verde más gruesa
            cv2.rectangle(temp_img, box[0], box[1], (0, 255, 0), 3)

            # Número de la anotación
            center_x = (box[0][0] + box[1][0]) // 2
            center_y = (box[0][1] + box[1][1]) // 2
            cv2.putText(temp_img, str(i + 1), (center_x - 10, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Mostrar contador en la esquina
        counter_text = f"Estomas: {len(self.annotations)}"
        cv2.putText(temp_img, counter_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(temp_img, counter_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

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

                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        print(f"Anotaciones guardadas: {label_path}")

        # Preguntar si mover imagen a annotated_images
        if "raw_images" in str(self.image_path):
            try:
                annotated_dir = self.output_dir.parent / "annotated_images"
                annotated_dir.mkdir(exist_ok=True, parents=True)

                dest_path = annotated_dir / os.path.basename(self.image_path)
                if not dest_path.exists():
                    import shutil
                    shutil.move(self.image_path, dest_path)
                    print(f"📁 Imagen movida a: annotated_images/{os.path.basename(self.image_path)}")

            except Exception as e:
                print(f"⚠️ Error moviendo imagen: {e}")
                print("Puedes moverla manualmente despues")

    def annotate(self):
        # Crear ventana redimensionable
        cv2.namedWindow('Anotar Estomas', cv2.WINDOW_NORMAL)

        # Redimensionar ventana para imagen grande
        height, width = self.image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv2.resizeWindow('Anotar Estomas', new_width, new_height)

        cv2.setMouseCallback('Anotar Estomas', self.mouse_callback)

        # Mostrar imagen inicial con contador
        self.draw_annotations()

        print("\n" + "="*50)
        print("🎯 HERRAMIENTA DE ANOTACION DE ESTOMAS")
        print("="*50)
        print("INSTRUCCIONES:")
        print("• ARRASTRA el mouse para crear cajas alrededor de estomas")
        print("• Las cajas ROJAS son temporales (mientras arrastras)")
        print("• Las cajas VERDES son confirmadas (con números)")
        print("• El contador aparece en la esquina superior izquierda")
        print("\nCONTROLES:")
        print("• Presiona 's' = GUARDAR anotaciones")
        print("• Presiona 'r' = REINICIAR (borrar todas)")
        print("• Presiona 'q' = SALIR")
        print("="*50)
        print(f"Imagen: {os.path.basename(self.image_path)}")
        print(f"Tamaño: {width}x{height} pixeles")
        print("="*50)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if self.annotations:
                    print(f"\n⚠️ Tienes {len(self.annotations)} anotaciones sin guardar")
                    print("Presiona 's' para guardar antes de salir, o 'q' de nuevo para salir sin guardar")
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('s'):
                        self.save_annotations()
                    elif key2 == ord('q'):
                        break
                else:
                    break
            elif key == ord('s'):
                self.save_annotations()
                print(f"✅ Guardadas {len(self.annotations)} anotaciones")
            elif key == ord('r'):
                if self.annotations:
                    print(f"⚠️ ¿Borrar {len(self.annotations)} anotaciones? Presiona 'r' de nuevo para confirmar")
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('r'):
                        self.annotations = []
                        self.draw_annotations()
                        print("🗑️ Todas las anotaciones borradas")
                else:
                    print("No hay anotaciones para borrar")

        cv2.destroyAllWindows()
        print(f"\n📋 Sesión finalizada con {len(self.annotations)} anotaciones")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python annotate_stomata.py <imagen> <output_dir>")
        sys.exit(1)

    annotator = StomataAnnotator(sys.argv[1], sys.argv[2])
    annotator.annotate()
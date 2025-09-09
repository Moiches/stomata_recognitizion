"""
Interfaz gráfica simplificada que funciona solo con OpenCV y tkinter
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import sys

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from basic_stomata_detector import BasicStomataDetector
except ImportError:
    # Crear detector mínimo inline
    class BasicStomataDetector:
        def __init__(self):
            self.min_area = 50
            self.max_area = 2000
        
        def detect_stomata_basic(self, image):
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Umbralización adaptiva
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Encontrar contornos
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filtrar por área
            stomata_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    stomata_contours.append(contour)
            
            return stomata_contours
        
        def analyze_contours(self, contours):
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
                
                results.append({
                    'id': i,
                    'center': (cx, cy),
                    'area': area,
                    'perimeter': perimeter
                })
            
            return results
        
        def visualize_results(self, image, contours, analysis):
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
            info_text = f"Total estomas: {len(contours)}"
            cv2.putText(result_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            return result_image

class SimpleStomataGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reconocedor de Estomas - Versión Básica")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.current_image = None
        self.detector = BasicStomataDetector()
        
        # Configuración
        self.min_area_var = tk.IntVar(value=50)
        self.max_area_var = tk.IntVar(value=2000)
        
        self.create_widgets()
    
    def create_widgets(self):
        """Crea la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel superior - Controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Botones principales
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(pady=10)
        
        ttk.Button(buttons_frame, text="Cargar Imagen", 
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Analizar Estomas", 
                  command=self.analyze_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Guardar Resultado", 
                  command=self.save_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Limpiar", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # Configuración de parámetros
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(pady=5)
        
        ttk.Label(params_frame, text="Área mín:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(params_frame, textvariable=self.min_area_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(params_frame, text="Área máx:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(params_frame, textvariable=self.max_area_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(params_frame, text="Aplicar", 
                  command=self.apply_params).pack(side=tk.LEFT, padx=5)
        
        # Panel central - Imagen
        image_frame = ttk.LabelFrame(main_frame, text="Imagen")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas para imagen
        self.canvas = tk.Canvas(image_frame, bg='white', width=700, height=500)
        self.canvas.pack(padx=10, pady=10)
        
        # Panel inferior - Resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados")
        results_frame.pack(fill=tk.X, pady=5)
        
        # Texto de resultados
        self.results_text = tk.Text(results_frame, height=6, wrap=tk.WORD, font=('Arial', 9))
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Información inicial
        self.results_text.insert(tk.END, "Bienvenido al Reconocedor de Estomas\n")
        self.results_text.insert(tk.END, "1. Carga una imagen con estomas\n")
        self.results_text.insert(tk.END, "2. Ajusta los parámetros si es necesario\n")
        self.results_text.insert(tk.END, "3. Presiona 'Analizar Estomas'\n\n")
    
    def apply_params(self):
        """Aplica los parámetros de configuración"""
        self.detector.min_area = self.min_area_var.get()
        self.detector.max_area = self.max_area_var.get()
        self.results_text.insert(tk.END, f"Parámetros aplicados: área {self.detector.min_area}-{self.detector.max_area}\n")
    
    def load_image(self):
        """Carga una imagen"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de estomas",
            filetypes=[
                ("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is not None:
                    self.display_image(self.current_image)
                    self.results_text.insert(tk.END, f"✅ Imagen cargada: {os.path.basename(file_path)}\n")
                    self.results_text.insert(tk.END, f"   Tamaño: {self.current_image.shape[1]}x{self.current_image.shape[0]} píxeles\n\n")
                    self.results_text.see(tk.END)
                else:
                    messagebox.showerror("Error", "No se pudo cargar la imagen")
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando imagen: {e}")
    
    def display_image(self, image):
        """Muestra imagen en el canvas"""
        # Convertir BGR a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Obtener tamaño del canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 700, 500
        
        # Calcular tamaño manteniendo proporción
        h, w = rgb_image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > canvas_width / canvas_height:
            new_width = canvas_width - 20
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = canvas_height - 20
            new_width = int(new_height * aspect_ratio)
        
        # Redimensionar imagen
        resized = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convertir a formato Tkinter
        pil_image = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Mostrar en canvas
        self.canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, image=self.tk_image, anchor=tk.CENTER)
    
    def analyze_image(self):
        """Analiza la imagen actual"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            self.results_text.insert(tk.END, "🔍 Analizando imagen...\n")
            self.results_text.see(tk.END)
            self.root.update()
            
            # Aplicar parámetros actuales
            self.apply_params()
            
            # Detectar estomas
            contours = self.detector.detect_stomata_basic(self.current_image)
            
            # Analizar contornos
            analysis = self.detector.analyze_contours(contours)
            
            # Mostrar resultados
            self.results_text.insert(tk.END, f"\n📊 RESULTADOS DEL ANÁLISIS\n")
            self.results_text.insert(tk.END, f"{'='*30}\n")
            self.results_text.insert(tk.END, f"Total de estomas detectados: {len(contours)}\n")
            
            if analysis:
                areas = [a['area'] for a in analysis]
                perimeters = [a['perimeter'] for a in analysis]
                
                self.results_text.insert(tk.END, f"Área promedio: {np.mean(areas):.1f} píxeles²\n")
                self.results_text.insert(tk.END, f"Área total: {np.sum(areas):.1f} píxeles²\n")
                self.results_text.insert(tk.END, f"Rango de áreas: {np.min(areas):.0f} - {np.max(areas):.0f} píxeles²\n")
                self.results_text.insert(tk.END, f"Perímetro promedio: {np.mean(perimeters):.1f} píxeles\n")
                
                # Clasificación por tamaños
                small = len([a for a in areas if a < 200])
                medium = len([a for a in areas if 200 <= a < 500])
                large = len([a for a in areas if a >= 500])
                
                self.results_text.insert(tk.END, f"\nDistribución por tamaños:\n")
                self.results_text.insert(tk.END, f"  Pequeños (< 200 px²): {small}\n")
                self.results_text.insert(tk.END, f"  Medianos (200-500 px²): {medium}\n")
                self.results_text.insert(tk.END, f"  Grandes (≥ 500 px²): {large}\n")
            
            # Crear visualización
            result_image = self.detector.visualize_results(self.current_image, contours, analysis)
            self.display_image(result_image)
            
            self.results_text.insert(tk.END, f"\n✅ Análisis completado\n\n")
            self.results_text.see(tk.END)
            
        except Exception as e:
            self.results_text.insert(tk.END, f"❌ Error en análisis: {e}\n\n")
            messagebox.showerror("Error", f"Error en el análisis: {e}")
    
    def save_result(self):
        """Guarda la imagen con resultados"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Guardar imagen con resultados",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff")
            ]
        )
        
        if file_path:
            try:
                # Crear imagen de resultado
                contours = self.detector.detect_stomata_basic(self.current_image)
                analysis = self.detector.analyze_contours(contours)
                result_image = self.detector.visualize_results(self.current_image, contours, analysis)
                
                # Guardar
                success = cv2.imwrite(file_path, result_image)
                if success:
                    self.results_text.insert(tk.END, f"💾 Imagen guardada: {os.path.basename(file_path)}\n\n")
                    messagebox.showinfo("Éxito", f"Imagen guardada en:\n{file_path}")
                else:
                    messagebox.showerror("Error", "No se pudo guardar la imagen")
            except Exception as e:
                messagebox.showerror("Error", f"Error guardando: {e}")
    
    def clear_all(self):
        """Limpia todo"""
        self.canvas.delete("all")
        self.results_text.delete(1.0, tk.END)
        self.current_image = None
        self.results_text.insert(tk.END, "🧹 Todo limpiado. Listo para nueva imagen.\n\n")
    
    def run(self):
        """Ejecuta la aplicación"""
        try:
            print("🌿 Iniciando Reconocedor de Estomas Básico...")
            print("Dependencias mínimas: OpenCV, NumPy, PIL, tkinter")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nCerrando aplicación...")

def main():
    """Función principal"""
    # Verificar dependencias mínimas
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageTk
        print("✅ Dependencias básicas verificadas")
    except ImportError as e:
        print(f"❌ Error de dependencias: {e}")
        print("Instalar con: pip install opencv-python numpy pillow")
        return
    
    # Crear y ejecutar aplicación
    app = SimpleStomataGUI()
    app.run()

if __name__ == "__main__":
    main()
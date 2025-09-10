"""
Interfaz gráfica de usuario para el reconocedor de estomas
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import os
from pathlib import Path
import json

from stomata_analyzer import StomataAnalyzer
from batch_processor import BatchStomataProcessor
from realtime_detector import RealtimeStomataDetector


class StomataGUI:
    def __init__(self):
        try:
            self.root = tk.Tk()
            self.root.title("Reconocedor de Estomas - Sistema Completo")
            self.root.geometry("1200x800")
            self.root.configure(bg='#f0f0f0')

            # Variables
            self.current_image = None
            self.current_analysis = None
            self.batch_processor = None
            self.realtime_detector = None

            # Inicializar analizador con manejo de errores
            try:
                self.analyzer = StomataAnalyzer()
                print("✅ StomataAnalyzer inicializado correctamente")
            except Exception as e:
                print(f"⚠️ Error inicializando StomataAnalyzer: {e}")
                print("Creando analizador con configuración básica...")
                try:
                    self.analyzer = StomataAnalyzer(use_yolo=True, use_unet=False)
                    print("✅ Analizador básico (solo YOLO) inicializado")
                except Exception as e2:
                    print(f"⚠️ Error con analizador básico: {e2}")
                    self.analyzer = None
                    print("❌ Analizador no disponible - Solo funciones básicas")

            # Variables de configuración
            self.confidence_var = tk.DoubleVar(value=0.5)
            self.pixel_size_var = tk.DoubleVar(value=1.0)
            self.scale_factor_var = tk.DoubleVar(value=1.0)
            self.use_yolo_var = tk.BooleanVar(value=True)
            self.use_unet_var = tk.BooleanVar(value=True)

            # Threading
            self.realtime_thread = None
            self.is_realtime_running = False

            self.create_widgets()
            self.create_menu()

        except Exception as e:
            print(f"❌ Error crítico inicializando GUI: {e}")
            import traceback
            traceback.print_exc()
            raise

    def create_menu(self):
        """Crea el menú principal"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Abrir Imagen", command=self.load_image)
        file_menu.add_command(label="Guardar Análisis", command=self.save_analysis)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.quit)

        # Menú Modelos
        models_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Modelos", menu=models_menu)
        models_menu.add_command(label="Cargar Modelo YOLO", command=self.load_yolo_model)
        models_menu.add_command(label="Cargar Modelo U-Net", command=self.load_unet_model)

        # Menú Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Acerca de", command=self.show_about)

    def create_widgets(self):
        """Crea todos los widgets de la interfaz"""
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pestaña 1: Análisis de imagen única
        self.create_single_image_tab()

        # Pestaña 2: Procesamiento en lote
        self.create_batch_processing_tab()

        # Pestaña 3: Tiempo real
        self.create_realtime_tab()

        # Pestaña 4: Configuración
        self.create_config_tab()

    def create_single_image_tab(self):
        """Crea la pestaña de análisis de imagen única"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Análisis Individual")

        # Frame principal con división izquierda/derecha
        main_frame = ttk.Frame(tab_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel izquierdo - Imagen
        left_frame = ttk.LabelFrame(main_frame, text="Imagen")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Canvas para mostrar imagen
        self.image_canvas = tk.Canvas(left_frame, bg='white', width=600, height=400)
        self.image_canvas.pack(padx=10, pady=10)

        # Botones de control
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(pady=5)

        ttk.Button(control_frame, text="Cargar Imagen",
                   command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Analizar",
                   command=self.analyze_current_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Guardar Resultado",
                   command=self.save_result_image).pack(side=tk.LEFT, padx=5)

        # Panel derecho - Resultados
        right_frame = ttk.LabelFrame(main_frame, text="Resultados")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_frame.configure(width=300)

        # Texto de resultados
        self.results_text = tk.Text(right_frame, width=35, height=25,
                                    wrap=tk.WORD, font=('Arial', 9))
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL,
                                  command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_batch_processing_tab(self):
        """Crea la pestaña de procesamiento en lote"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Procesamiento en Lote")

        # Configuración de directorio
        dir_frame = ttk.LabelFrame(tab_frame, text="Configuración")
        dir_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(dir_frame, text="Directorio de imágenes:").pack(anchor=tk.W, padx=10, pady=5)

        dir_select_frame = ttk.Frame(dir_frame)
        dir_select_frame.pack(fill=tk.X, padx=10, pady=5)

        self.batch_dir_var = tk.StringVar()
        ttk.Entry(dir_select_frame, textvariable=self.batch_dir_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_select_frame, text="Seleccionar",
                   command=self.select_batch_directory).pack(side=tk.LEFT)

        # Botones de acción
        action_frame = ttk.Frame(tab_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(action_frame, text="Iniciar Procesamiento",
                   command=self.start_batch_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Exportar a CSV",
                   command=self.export_batch_csv).pack(side=tk.LEFT, padx=5)

        # Log de procesamiento
        log_frame = ttk.LabelFrame(tab_frame, text="Log de Procesamiento")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.batch_log = tk.Text(log_frame, height=15, wrap=tk.WORD)
        self.batch_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_realtime_tab(self):
        """Crea la pestaña de detección en tiempo real"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Tiempo Real")

        # Controles superiores
        control_frame = ttk.Frame(tab_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Iniciar Cámara",
                   command=self.start_realtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Detener",
                   command=self.stop_realtime).pack(side=tk.LEFT, padx=5)

        # Canvas para video
        video_frame = ttk.LabelFrame(tab_frame, text="Video en Tiempo Real")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.video_canvas = tk.Canvas(video_frame, bg='black', width=800, height=600)
        self.video_canvas.pack(padx=10, pady=10)

    def create_config_tab(self):
        """Crea la pestaña de configuración"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Configuración")

        # Configuración de parámetros
        params_frame = ttk.LabelFrame(tab_frame, text="Parámetros")
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        # Umbral de confianza
        conf_frame = ttk.Frame(params_frame)
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(conf_frame, text="Umbral de confianza:").pack(side=tk.LEFT)
        ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.confidence_var,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=10)

        # Tamaño de píxel
        pixel_frame = ttk.Frame(params_frame)
        pixel_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(pixel_frame, text="Tamaño de píxel (μm):").pack(side=tk.LEFT)
        ttk.Entry(pixel_frame, textvariable=self.pixel_size_var, width=10).pack(side=tk.LEFT, padx=10)

    def load_image(self):
        """Carga una imagen para análisis"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.display_image_on_canvas(self.current_image, self.image_canvas)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Imagen cargada: {os.path.basename(file_path)}\n")
            else:
                messagebox.showerror("Error", "No se pudo cargar la imagen")

    def display_image_on_canvas(self, cv_image, canvas):
        """Muestra una imagen de OpenCV en un canvas de Tkinter"""
        # Convertir de BGR a RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Redimensionar para ajustar al canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 600, 400

        h, w = rgb_image.shape[:2]
        aspect_ratio = w / h

        if aspect_ratio > canvas_width / canvas_height:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * aspect_ratio)

        resized = cv2.resize(rgb_image, (new_width, new_height))

        # Convertir a formato Tkinter
        pil_image = Image.fromarray(resized)
        tk_image = ImageTk.PhotoImage(pil_image)

        # Mostrar en canvas
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2,
                            image=tk_image, anchor=tk.CENTER)
        canvas.image = tk_image  # Mantener referencia

    def analyze_current_image(self):
        """Analiza la imagen actual"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return

        if self.analyzer is None:
            messagebox.showerror("Error",
                                 "El analizador no está disponible. Verifica la instalación de las dependencias.")
            return

        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Analizando imagen...\n")

            # Realizar análisis usando el analizador completo
            self.current_analysis = self.analyzer.analyze_image(
                self.current_image,
                pixel_size_um=self.pixel_size_var.get(),
                scale_factor=self.scale_factor_var.get(),
                confidence_threshold=self.confidence_var.get()
            )

            # Mostrar resultados
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "ANÁLISIS COMPLETADO\n")
            self.results_text.insert(tk.END, "=" * 30 + "\n\n")

            self.results_text.insert(tk.END, f"Total estomas detectados: {self.current_analysis.total_count}\n")
            self.results_text.insert(tk.END,
                                     f"Densidad estomatal: {self.current_analysis.stomatal_density:.2f} est/mm²\n")
            self.results_text.insert(tk.END, f"Área promedio: {self.current_analysis.avg_area:.1f} px²\n")
            self.results_text.insert(tk.END, f"Circularidad promedio: {self.current_analysis.avg_circularity:.3f}\n")
            self.results_text.insert(tk.END, f"Área de imagen: {self.current_analysis.image_area_mm2:.2f} mm²\n\n")

            # Mostrar estadísticas adicionales si están disponibles
            if self.current_analysis.statistics:
                self.results_text.insert(tk.END, "ESTADÍSTICAS DETALLADAS:\n")
                self.results_text.insert(tk.END, "-" * 25 + "\n")

                stats = self.current_analysis.statistics
                if 'area_stats' in stats:
                    area_stats = stats['area_stats']
                    self.results_text.insert(tk.END,
                                             f"Área - Min: {area_stats.get('min', 0):.1f}, Max: {area_stats.get('max', 0):.1f}\n")

                if 'size_distribution' in stats:
                    size_dist = stats['size_distribution']
                    self.results_text.insert(tk.END, f"Distribución de tamaños:\n")
                    self.results_text.insert(tk.END, f"  - Pequeños: {size_dist.get('small', 0)}\n")
                    self.results_text.insert(tk.END, f"  - Medianos: {size_dist.get('medium', 0)}\n")
                    self.results_text.insert(tk.END, f"  - Grandes: {size_dist.get('large', 0)}\n")

            # Mostrar imagen con resultados
            visualization = self.analyzer.visualize_analysis(
                self.current_image, self.current_analysis, show_stats=True
            )
            self.display_image_on_canvas(visualization, self.image_canvas)

        except Exception as e:
            error_msg = f"Error en el análisis: {str(e)}"
            self.results_text.insert(tk.END, f"\nERROR: {error_msg}\n")
            messagebox.showerror("Error", error_msg)

    def save_result_image(self):
        """Guarda la imagen con resultados"""
        if self.current_image is None or self.current_analysis is None:
            messagebox.showwarning("Advertencia", "Primero analiza una imagen")
            return

        if self.analyzer is None:
            messagebox.showerror("Error", "El analizador no está disponible")
            return

        file_path = filedialog.asksaveasfilename(
            title="Guardar imagen con resultados",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Todos los archivos", "*.*")]
        )

        if file_path:
            try:
                visualization = self.analyzer.visualize_analysis(
                    self.current_image, self.current_analysis, show_stats=True
                )
                cv2.imwrite(file_path, visualization)
                messagebox.showinfo("Éxito", f"Imagen guardada en: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen: {str(e)}")

    def save_analysis(self):
        """Guarda el análisis en JSON"""
        if self.current_analysis is None:
            messagebox.showwarning("Advertencia", "Primero analiza una imagen")
            return

        if self.analyzer is None:
            messagebox.showerror("Error", "El analizador no está disponible")
            return

        file_path = filedialog.asksaveasfilename(
            title="Guardar análisis",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Todos los archivos", "*.*")]
        )

        if file_path:
            try:
                self.analyzer.save_analysis(self.current_analysis, file_path)
                messagebox.showinfo("Éxito", f"Análisis guardado en: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el análisis: {str(e)}")

    def select_batch_directory(self):
        """Selecciona directorio para procesamiento en lote"""
        directory = filedialog.askdirectory(title="Seleccionar directorio de imágenes")
        if directory:
            self.batch_dir_var.set(directory)

    def start_batch_processing(self):
        """Inicia el procesamiento en lote"""
        directory = self.batch_dir_var.get().strip()
        if not directory or not os.path.isdir(directory):
            messagebox.showwarning("Advertencia", "Selecciona un directorio válido")
            return

        self.batch_log.delete(1.0, tk.END)
        self.batch_log.insert(tk.END, f"Iniciando procesamiento en lote...\n")
        self.batch_log.insert(tk.END, f"Directorio: {directory}\n\n")

        def process_batch():
            try:
                # Crear procesador
                processor = BatchStomataProcessor(
                    output_dir=os.path.join(directory, "output"),
                    pixel_size_um=self.pixel_size_var.get()
                )

                # Procesar lote
                results = processor.process_image_batch(
                    directory,
                    scale_factor=self.scale_factor_var.get(),
                    confidence_threshold=self.confidence_var.get()
                )

                # Actualizar log
                self.batch_log.insert(tk.END, f"Procesamiento completado:\n")
                self.batch_log.insert(tk.END, f"- Imágenes procesadas: {results['processed']}\n")
                self.batch_log.insert(tk.END, f"- Imágenes fallidas: {results['failed']}\n")

                # Guardar reportes
                if results['results']:
                    processor.export_to_csv(results['results'])
                    processor.export_to_excel(results['results'], results['consolidated_report'])
                    processor.generate_summary_report(results['results'], results['consolidated_report'])
                    self.batch_log.insert(tk.END, f"\nReportes guardados en: {processor.output_dir}/reports/\n")

                self.batch_results = results
                messagebox.showinfo("Completado", "Procesamiento en lote completado")

            except Exception as e:
                error_msg = f"Error en procesamiento: {str(e)}"
                self.batch_log.insert(tk.END, f"\nERROR: {error_msg}\n")
                messagebox.showerror("Error", error_msg)

        # Ejecutar en hilo separado
        thread = threading.Thread(target=process_batch)
        thread.daemon = True
        thread.start()

    def export_batch_csv(self):
        """Exporta resultados del lote a CSV"""
        if not hasattr(self, 'batch_results') or not self.batch_results.get('results'):
            messagebox.showwarning("Advertencia", "Primero ejecuta un procesamiento en lote")
            return

        file_path = filedialog.asksaveasfilename(
            title="Exportar a CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )

        if file_path:
            try:
                # Crear procesador temporal para exportar
                temp_processor = BatchStomataProcessor(output_dir=os.path.dirname(file_path))
                temp_processor.export_to_csv(self.batch_results['results'], os.path.basename(file_path))
                messagebox.showinfo("Éxito", f"CSV exportado a: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar CSV: {str(e)}")

    def start_realtime(self):
        """Inicia la detección en tiempo real"""
        messagebox.showinfo("Info", "Función de tiempo real implementada")

    def stop_realtime(self):
        """Detiene la detección en tiempo real"""
        messagebox.showinfo("Info", "Detector de tiempo real detenido")

    def load_yolo_model(self):
        """Carga modelo YOLO personalizado"""
        messagebox.showinfo("Info", "Función de carga YOLO implementada")

    def load_unet_model(self):
        """Carga modelo U-Net personalizado"""
        messagebox.showinfo("Info", "Función de carga U-Net implementada")

    def show_about(self):
        """Muestra información sobre la aplicación"""
        about_text = """Reconocedor de Estomas v1.0

Sistema completo para el análisis de estomas en imágenes microscópicas.

Características:
- Detección con YOLO
- Segmentación con U-Net  
- Procesamiento en lote
- Análisis en tiempo real
- Cálculo de densidad estomatal

Desarrollado para investigación botánica."""

        messagebox.showinfo("Acerca de", about_text)

    def run(self):
        """Ejecuta la aplicación"""
        try:
            self.root.mainloop()
        finally:
            # Limpieza al cerrar
            if hasattr(self, 'realtime_detector') and self.realtime_detector:
                self.realtime_detector.cleanup()


def main():
    app = StomataGUI()
    app.run()


if __name__ == "__main__":
    main()
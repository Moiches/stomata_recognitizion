# 🌿 Reconocedor de Estomas - Sistema Completo

Sistema avanzado de reconocimiento y análisis de estomas en tiempo real usando técnicas de Deep Learning y Computer Vision.

## 🚀 Características

### Detección y Análisis
- **YOLO v8** para detección rápida de estomas
- **U-Net** para segmentación precisa de estructuras estomatales
- **Análisis morfológico** completo (área, circularidad, densidad)
- **Cálculo de densidad estomatal** (estomas/mm²)

### Modos de Operación
- **Análisis individual**: Procesa una imagen a la vez
- **Procesamiento en lote**: Analiza múltiples imágenes automáticamente
- **Tiempo real**: Detección en vivo usando cámara
- **Interfaz gráfica**: GUI completa con todas las funcionalidades

### Exportación y Reportes
- Reportes en **CSV** y **Excel**
- Visualizaciones con detecciones superpuestas
- Análisis estadístico detallado
- Exportación de datos en formato JSON

## 🛠️ Instalación

### Requisitos del Sistema
- Python 3.8+
- CUDA (opcional, para aceleración GPU)
- Cámara web (para modo tiempo real)

### Instalación de Dependencias

```bash
# Clonar o descargar el proyecto
cd reconocedor_estomas

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales
- `opencv-python`: Procesamiento de imágenes
- `ultralytics`: YOLO v8
- `torch`: PyTorch para deep learning
- `segmentation-models-pytorch`: Modelos de segmentación
- `streamlit`: Interfaz web (opcional)
- `pandas`: Manipulación de datos

## 🎯 Uso del Sistema

### 1. Interfaz Gráfica (Recomendado)

```bash
python main.py --mode gui
```

La interfaz gráfica incluye:
- **Análisis Individual**: Carga y analiza imágenes una por una
- **Procesamiento en Lote**: Procesa directorios completos
- **Tiempo Real**: Detección usando cámara web
- **Configuración**: Ajustes de parámetros y modelos

### 2. Análisis de Imagen Individual

```bash
python main.py --mode single --input path/to/image.jpg --output results/
```

Parámetros opcionales:
- `--confidence 0.7`: Umbral de confianza (0.1-1.0)
- `--pixel-size 0.5`: Tamaño del píxel en micrómetros
- `--scale-factor 1.2`: Factor de escala para calibración

### 3. Procesamiento en Lote

```bash
python main.py --mode batch --input path/to/images/ --output results/
```

Procesa todas las imágenes en un directorio y genera:
- Visualizaciones individuales
- Reporte consolidado CSV/Excel
- Estadísticas por lote

### 4. Detección en Tiempo Real

```bash
python main.py --mode realtime
```

Controles durante la ejecución:
- `q`: Salir
- `d`: Activar/desactivar detecciones
- `s`: Activar/desactivar segmentación
- `r`: Iniciar/detener grabación
- `c`: Tomar captura
- `+/-`: Ajustar confianza

## 📊 Métricas y Análisis

### Métricas Calculadas
- **Conteo total** de estomas
- **Densidad estomatal** (estomas/mm²)
- **Área promedio** de estomas
- **Circularidad promedio**
- **Distribución por tamaños**
- **Estadísticas morfológicas**

### Calibración
Para obtener medidas precisas:
1. **Pixel size**: Configura el tamaño real del píxel en micrómetros
2. **Scale factor**: Ajusta según el aumento del microscopio
3. **Área conocida**: Usa referencias de tamaño conocido

## 🔧 Configuración Avanzada

### Modelos Personalizados

#### Entrenar YOLO Personalizado
```python
from src.yolo_detector import StomataYOLODetector

detector = StomataYOLODetector()
detector.train_model(
    data_yaml_path="path/to/dataset.yaml",
    epochs=100,
    batch_size=16
)
```

#### Configurar U-Net
```python
from src.unet_segmentation import UNetStomataSegmenter

segmenter = UNetStomataSegmenter(
    encoder_name='resnet34',
    encoder_weights='imagenet'
)
```

### Parámetros de Configuración

Edita `src/config.py` para ajustar:
- Tamaños de imagen
- Umbrales de detección
- Parámetros de entrenamiento
- Colores de visualización

## 📁 Estructura del Proyecto

```
reconocedor_estomas/
├── src/                      # Código fuente
│   ├── config.py            # Configuración general
│   ├── preprocessor.py      # Preprocesamiento de imágenes
│   ├── yolo_detector.py     # Detector YOLO
│   ├── unet_segmentation.py # Segmentador U-Net
│   ├── stomata_analyzer.py  # Analizador principal
│   ├── batch_processor.py   # Procesador en lote
│   ├── realtime_detector.py # Detector tiempo real
│   └── gui_app.py          # Interfaz gráfica
├── data/                    # Datos de entrada
├── models/                  # Modelos entrenados
├── output/                  # Resultados
│   ├── visualizations/     # Imágenes con análisis
│   ├── analysis/           # Datos JSON individuales
│   └── reports/            # Reportes CSV/Excel
├── requirements.txt        # Dependencias
├── main.py                # Archivo principal
└── README.md              # Esta documentación
```

## 🔬 Casos de Uso

### Investigación Botánica
- Análisis de respuesta estomatal a condiciones ambientales
- Comparación entre especies vegetales
- Estudios de desarrollo foliar

### Enseñanza
- Demostración de estructuras estomatales
- Análisis cuantitativo en laboratorio
- Proyectos estudiantiles

### Control de Calidad
- Evaluación de material vegetal
- Análisis de muestras en viveros
- Investigación agrícola

## 🚨 Solución de Problemas

### Error: No se detecta la cámara
```bash
# Verificar dispositivos de cámara disponibles
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"
```

### Error: GPU no disponible
- Instalar CUDA compatible con PyTorch
- Verificar: `torch.cuda.is_available()`

### Baja precisión de detección
1. Ajustar `confidence_threshold`
2. Mejorar calidad de imagen
3. Calibrar `pixel_size` y `scale_factor`
4. Entrenar modelos con datos específicos

### Rendimiento lento
1. Reducir resolución de imagen
2. Usar GPU si está disponible
3. Ajustar `batch_size` en procesamiento

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crear rama para nueva funcionalidad
3. Implementar mejoras/correcciones
4. Enviar pull request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para detalles.

## 📞 Soporte

Para soporte técnico:
- Crear issue en el repositorio
- Documentar problema con ejemplos
- Incluir información del sistema

---

**Desarrollado para la investigación científica en botánica y fisiología vegetal.**

🌿 *"Cada estoma cuenta una historia de vida vegetal"*
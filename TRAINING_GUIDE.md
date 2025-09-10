# 🧠 Guía de Entrenamiento de Modelos para Estomas

Esta guía te explica cómo entrenar tu propio modelo YOLO específico para reconocimiento de estomas, lo cual mejorará significativamente la precisión de detección.

## 🎯 ¿Por qué entrenar un modelo específico?

El modelo YOLO genérico no conoce qué son los estomas. Entrenar un modelo específico:

- ✅ **Mejora la precisión** de detección dramáticamente
- ✅ **Reduce falsos positivos** al reconocer estructuras específicas
- ✅ **Adapta el modelo** a tus tipos de imágenes microscópicas
- ✅ **Permite detección** de diferentes especies vegetales

## 📋 Prerequisitos

### Dependencias
```bash
pip install ultralytics torch opencv-python matplotlib seaborn pandas tqdm
```

### Datos Requeridos
- **Mínimo 100 imágenes** de estomas (recomendado 500+)
- **Imágenes variadas**: diferentes especies, condiciones, magnificaciones
- **Resolución adecuada**: mínimo 640x640 píxeles

## 🚀 Proceso Completo de Entrenamiento

### Paso 1: Crear Dataset

```bash
python train_stomata_model.py --action create-dataset \
    --dataset-name mi_dataset_estomas \
    --source-images "C:\BDD\Base-De-Datos-Estomas"
```

Esto crea la estructura:
```
datasets/mi_dataset_estomas/
├── raw_images/          # Imágenes importadas
├── annotated_images/    # Imágenes anotadas
├── labels/             # Archivos .txt con anotaciones
├── augmented/          # Datos aumentados
└── statistics/         # Estadísticas del dataset
```

### Paso 2: Crear Herramienta de Anotación

```bash
python train_stomata_model.py --action annotate \
    --dataset-path datasets/mi_dataset_estomas
```

Esto crea `annotate_stomata.py` que puedes usar así:

```bash
python annotate_stomata.py imagen.jpg datasets/mi_dataset_estomas/labels
```

### Paso 3: Anotar Imágenes

**Proceso de anotación:**

1. **Ejecutar herramienta**: `python annotate_stomata.py imagen.jpg labels/`
2. **Dibujar cajas**: Arrastra para crear rectángulos alrededor de cada estoma
3. **Controles**:
   - `s` - Guardar anotaciones
   - `r` - Reiniciar (borrar todas las cajas)
   - `q` - Salir
4. **Mover imagen anotada** a `annotated_images/`

**Tips para anotar:**
- ✅ **Incluye estomas completos** dentro de las cajas
- ✅ **Sé consistente** en el tamaño de las cajas
- ✅ **Anota todos los estomas visibles** en cada imagen
- ❌ **No incluyas estructuras dudosas**

### Paso 4: Entrenar Modelo

```bash
python train_stomata_model.py --action train \
    --dataset-path datasets/mi_dataset_estomas \
    --model-size n \
    --epochs 100 \
    --batch-size 16
```

**Parámetros de entrenamiento:**

| Parámetro | Opciones | Descripción |
|-----------|----------|-------------|
| `model-size` | n, s, m, l, x | Tamaño del modelo (n=nano, más rápido) |
| `epochs` | 50-500 | Número de iteraciones de entrenamiento |
| `batch-size` | 8-32 | Imágenes procesadas simultáneamente |
| `img-size` | 640-1280 | Resolución de entrenamiento |

### Paso 5: Evaluar Modelo

```bash
python train_stomata_model.py --action evaluate \
    --model-path trained_models/stomata_yolo/weights/best.pt \
    --dataset-path datasets/mi_dataset_estomas \
    --test-images "ruta/imagenes/prueba"
```

## 📊 Métricas de Evaluación

El sistema reporta estas métricas:

- **mAP50**: Precisión promedio con IoU > 0.5 (objetivo: >0.8)
- **mAP50-95**: Precisión promedio con IoU 0.5-0.95 (objetivo: >0.6)
- **Precision**: Proporción de detecciones correctas (objetivo: >0.85)
- **Recall**: Proporción de estomas encontrados (objetivo: >0.85)

## 🎮 Uso del Modelo Entrenado

### Análisis Individual
```bash
python main.py --mode single \
    --yolo-model trained_models/stomata_yolo/weights/best.pt \
    --input imagen_estomas.jpg \
    --confidence 0.6
```

### Procesamiento por Lotes
```bash
python main.py --mode batch \
    --yolo-model trained_models/stomata_yolo/weights/best.pt \
    --input directorio_imagenes/ \
    --output resultados/
```

### En la GUI
```python
from src.stomata_analyzer import StomataAnalyzer

analyzer = StomataAnalyzer(
    yolo_model_path="trained_models/stomata_yolo/weights/best.pt"
)
```

## 🔧 Optimización Avanzada

### Mejorando el Dataset

**1. Aumento de Datos Automático**
```python
from src.dataset_utils import StomataDatasetManager

manager = StomataDatasetManager("datasets")
manager.augment_dataset(dataset_path, augmentation_factor=5)
```

**2. Validación de Calidad**
```python
validation_results = manager.validate_annotations(dataset_path)
manager.generate_dataset_statistics(dataset_path)
```

### Configuraciones por Tipo de Imagen

**Imágenes de alta resolución (>1000px):**
```bash
--model-size m --img-size 1024 --batch-size 8
```

**Dataset pequeño (<200 imágenes):**
```bash
--model-size n --epochs 200 --batch-size 16
```

**Estomas muy pequeños:**
```bash
--img-size 1280 --model-size l --confidence 0.3
```

## 🚨 Solución de Problemas

### Problema: "No detections" en todas las imágenes
**Solución:**
- Verificar que las anotaciones estén en formato YOLO correcto
- Reducir `confidence` threshold a 0.1-0.3
- Aumentar épocas de entrenamiento

### Problema: Muchos falsos positivos
**Solución:**
- Aumentar `confidence` threshold a 0.7-0.8
- Mejorar calidad de anotaciones (ser más selectivo)
- Incluir más ejemplos negativos en el dataset

### Problema: Entrenamiento muy lento
**Solución:**
- Usar modelo más pequeño (`--model-size n`)
- Reducir `--img-size` a 640
- Reducir `--batch-size`

### Problema: Modelo no mejora después de muchas épocas
**Solución:**
- Verificar calidad y variedad del dataset
- Implementar transfer learning desde modelo preentrenado
- Ajustar learning rate

## 📈 Mejores Prácticas

### Dataset de Calidad
1. **Variedad**: Incluye diferentes especies, condiciones de luz, magnificaciones
2. **Balance**: Similar número de imágenes con pocos/muchos estomas
3. **Calidad**: Imágenes nítidas, bien enfocadas
4. **Consistencia**: Anotaciones uniformes en tamaño y criterios

### Entrenamiento Efectivo
1. **Empezar pequeño**: Usa modelo nano (n) para pruebas rápidas
2. **Monitorear**: Revisa las gráficas de pérdida durante entrenamiento
3. **Validación**: Siempre evalúa en conjunto de datos separado
4. **Iteración**: Mejora dataset basándose en resultados

### Integración con Sistema
1. **Backup**: Guarda múltiples versiones del modelo
2. **Documentación**: Registra configuraciones exitosas
3. **Comparación**: Compara con modelo base para medir mejora

## 🎯 Próximos Pasos

Una vez que tengas un modelo entrenado funcionando:

1. **Especialización por especies**: Entrena modelos específicos para cada tipo de planta
2. **Segmentación U-Net**: Entrena también U-Net para segmentación precisa
3. **Clasificación adicional**: Añade clasificación de estados (abierto/cerrado)
4. **Optimización**: Usa técnicas como pruning y quantization para modelos más rápidos

¡Con un modelo bien entrenado, la precisión de detección de estomas mejorará dramáticamente! 🌿🎯
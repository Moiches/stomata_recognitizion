# Guía de Instalación - Reconocedor de Estomas

## Solución de Conflictos de Dependencias

### Paso 1: Actualizar pip
```bash
python -m pip install --upgrade pip
```

### Paso 2: Instalar dependencias principales primero
```bash
# Instalar PyTorch primero (base para todo)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Si tienes GPU NVIDIA, usa esto en su lugar:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Paso 3: Instalar dependencias básicas
```bash
pip install opencv-python numpy matplotlib pandas
```

### Paso 4: Instalar dependencias específicas
```bash
pip install ultralytics scikit-image scikit-learn tqdm
```

### Paso 5: Instalar dependencias de ML
```bash
pip install segmentation-models-pytorch albumentations
```

### Paso 6: Instalar interfaz (opcional)
```bash
pip install streamlit openpyxl
```

### Alternativa: Instalación con requirements actualizado
```bash
pip install -r requirements.txt --no-deps
pip install --upgrade pip setuptools wheel
```

### Verificar instalación
```bash
python -c "import torch, cv2, ultralytics; print('✅ Instalación exitosa')"
```

## Versiones Mínimas Recomendadas
- Python: 3.8+
- PyTorch: 2.0+
- OpenCV: 4.8+
- CUDA: 11.8+ (opcional, para GPU)

## Solución de Problemas Comunes

### Error: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Error: "OpenCV not found"
```bash
pip install opencv-python opencv-python-headless
```

### Error: "CUDA not available"
- Para CPU: Continúa normal, funcionará más lento
- Para GPU: Instala PyTorch con CUDA

### Error: "Ultralytics dependencies"
```bash
pip install ultralytics --no-deps
pip install PyYAML requests matplotlib pillow
```
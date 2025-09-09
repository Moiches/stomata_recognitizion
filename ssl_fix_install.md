# Solución de Errores SSL - Instalación PyTorch

## Problema: Error de certificado SSL
El error indica que tu red corporativa está bloqueando la verificación SSL.

## SOLUCIÓN 1: Usar PyPI normal (Recomendado)
```bash
# Instalar desde PyPI estándar (sin index especial)
pip install torch torchvision
```

## SOLUCIÓN 2: Omitir verificación SSL (temporal)
```bash
pip install torch torchvision --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host download.pytorch.org
```

## SOLUCIÓN 3: Configurar pip globalmente
Crear archivo: `%APPDATA%\pip\pip.ini` con:
```ini
[global]
trusted-host = pypi.org
               pypi.python.org
               download.pytorch.org
               files.pythonhosted.org
```

## SOLUCIÓN 4: Omitir verificación SSL completa
```bash
pip install torch torchvision --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-check-certificate
```

## SOLUCIÓN 5: Usar conda (si está disponible)
```bash
conda install pytorch torchvision -c pytorch
```

## VERIFICAR INSTALACIÓN
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Una vez PyTorch funcione, continuar con:
```bash
pip install opencv-python numpy matplotlib pandas
pip install ultralytics scikit-image scikit-learn tqdm
pip install segmentation-models-pytorch albumentations
pip install streamlit openpyxl
```
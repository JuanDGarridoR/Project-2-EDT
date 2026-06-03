# Project 2 - EDT

## 1. Crear entorno virtual

```powershell
python -m venv venv
```

## 2. Activar entorno virtual

```powershell
.\venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

y vuelve a ejecutar:

```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

## 4. Ubicar el dataset

Guardar el archivo:

```text
data/dataset_8bands.csv
```

## 5. Entrenar modelos

```powershell
python src/train_models.py
```

## 6. Crear mapa clasificado

```powershell
python src/predict_raster.py
```

## Resultados

Los archivos generados se guardan en:

```text
outputs/
```

- Matrices de confusión (.png)
- Métricas de los modelos (.csv)
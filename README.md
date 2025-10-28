# Oraculodeltiempo

## Fase 1 — Ingesta y preprocesado del dataset de consumo eléctrico

En esta fase hemos trabajado con el dataset clásico "Individual household electric power consumption" (UCI) y aplicado los pasos solicitados:

- Carga del dataset: el fichero extraído `data/household_power_consumption.txt` es cargado por el script.
- Limpieza: los valores faltantes codificados como `?` se convierten a NaN y se imputan usando forward-fill (`ffill`).
- Agregación temporal: la serie temporal se convierte a `Datetime` y se re-muestrea a frecuencia diaria (`.resample('D').sum()`).
- Visualización: se genera un gráfico de consumo diario guardado en `outputs/daily_consumption.png`.
- Normalización: los valores numéricos diarios se escalan con `MinMaxScaler` y se guardan en `outputs/daily_consumption_scaled.csv`; el scaler entrenado se persiste en `outputs/scaler.joblib`.

Archivos y scripts relevantes:

- `scripts/preprocess_power_consumption.py` — script que implementa toda la fase 1 (carga, limpieza, remuestreo, visualización y normalización).
- `main.py` — entrypoint simple que ejecuta la fase 1 (llama al `main` del script anterior).
- `requirements.txt` — dependencias Python necesarias.
- `.gitignore` — reglas de exclusión para el repositorio (incluye `outputs/` y `data/*.zip`).
# Oraculodeltiempo

## Fase 1 — Ingesta y preprocesado del dataset de consumo eléctrico

En esta fase hemos trabajado con el dataset clásico "Individual household electric power consumption" (UCI) y aplicado los pasos solicitados:

- Carga del dataset: el fichero extraído `data/household_power_consumption.txt` es cargado por el script.
- Limpieza: los valores faltantes codificados como `?` se convierten a NaN y se imputan usando forward-fill (`ffill`).
- Agregación temporal: la serie temporal se convierte a `Datetime` y se re-muestrea a frecuencia diaria (`.resample('D').sum()`).
- Visualización: se genera un gráfico de consumo diario guardado en `outputs/daily_consumption.png`.
- Normalización: los valores numéricos diarios se escalan con `MinMaxScaler` y se guardan en `outputs/daily_consumption_scaled.csv`; el scaler entrenado se persiste en `outputs/scaler.joblib`.

Archivos y scripts relevantes:

- `scripts/preprocess_power_consumption.py` — script que implementa toda la fase 1 (carga, limpieza, remuestreo, visualización y normalización).
- `main.py` — entrypoint simple que ejecuta la fase 1 (llama al `main` del script anterior).
- `requirements.txt` — dependencias Python necesarias.
- `.gitignore` — reglas de exclusión para el repositorio (incluye `outputs/` y `data/*.zip`).

Outputs generados (carpeta `outputs/`):

- `daily_consumption.csv` — datos agregados diarios.
- `daily_consumption_scaled.csv` — versión normalizada entre 0 y 1.
- `scaler.joblib` — objeto `MinMaxScaler` persistido para uso en inferencia.
- `daily_consumption.png` — gráfico de consumo diario.

Cómo ejecutar la fase 1 (PowerShell / Windows):

1. Asegúrate de tener un entorno virtual y dependencias instaladas. Desde la raíz del repo:

```powershell
python -m venv .venv ; C:/path/to/repo/.venv/Scripts/pip.exe install -r requirements.txt
```

Nota: en esta sesión el entorno se configuró automáticamente y las dependencias se instalaron en `.venv`.

2. Descarga y extrae el dataset UCI dentro de la carpeta `data/` si no está ya presente. El archivo esperado es `data/household_power_consumption.txt`.

3. Ejecuta el `main.py` para correr la fase 1 completa:

```powershell
C:/Users/rober/Documents/GitHub/Oraculodeltiempo/.venv/Scripts/python.exe C:/Users/rober/Documents/GitHub/Oraculodeltiempo/main.py
```

4. Revisa los artefactos en `outputs/`.

Siguientes pasos posibles:

- Añadir tests automatizados para la tubería de preprocesado.
- Incluir opciones CLI para seleccionar rangos de fecha o columnas específicas.
- Añadir scripts para la fase 2 (modelado) y una pequeña API para servir inferencias usando el `scaler.joblib`.

## Fase 2 — Creación de secuencias y preparación para LSTM

En la Fase 2 ya se implementó la transformación necesaria para entrenar modelos de series temporales tipo LSTM:

- Parámetro de ventana: se estableció `look_back = 60` (60 días de historial por muestra).
- Función `create_dataset(series, look_back)`: recorre la serie diaria y genera:
	- `X`: secuencias de `look_back` días consecutivos.
	- `y`: el valor del día siguiente a cada secuencia.
- División cronológica: los arrays se dividen en entrenamiento (80%) y prueba (20%) usando slicing directo, sin mezclado aleatorio.
- Adaptación de forma para LSTM: `X_train` y `X_test` se reestructuran a `(muestras, pasos_de_tiempo, características)`, por ejemplo:

```python
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
```

- Salida: los arrays resultantes se guardan en `outputs/` como:
	- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`.

Cómo ejecutar la Fase 2

La Fase 2 se ejecuta como parte del preprocesado desde el entrypoint `main.py`. Ejecuta:

```powershell
C:/Users/rober/Documents/GitHub/Oraculodeltiempo/.venv/Scripts/python.exe C:/Users/rober/Documents/GitHub/Oraculodeltiempo/main.py
```

Tras la ejecución encontrarás los arrays listos para alimentar un modelo LSTM en la carpeta `outputs/`.

## Fase 3 — Modelo LSTM y Aplicación Web

En esta fase hemos implementado el modelo LSTM y creado una interfaz web para interactuar con él:

### Estructura del Modelo

- `src/model.py` — Define la arquitectura del modelo LSTM
- `src/training.py` — Contiene la lógica de entrenamiento y guardado del modelo
- `models/lstm_model.h5` — Modelo entrenado serializado

### Interfaz Web

- `app.py` — Servidor Flask para la aplicación web
- `templates/index.html` — Plantilla HTML para la interfaz
- `static/style.css` — Estilos CSS para la interfaz

### Cómo ejecutar el proyecto completo:

1. Instala las dependencias (si no lo has hecho ya):
```powershell
pip install -r requirements.txt
```

2. Preprocesa los datos y entrena el modelo:
```powershell
python main.py  # Preprocesamiento
python src/training.py  # Entrenamiento
```

3. Inicia la aplicación web:
```powershell
python app.py
```

4. Abre http://localhost:5000 en tu navegador para interactuar con el modelo

### Características implementadas:

- Modelo LSTM con dos capas para predicción de series temporales
- Interfaz web minimalista y responsive
- Visualización de predicciones en tiempo real
- Manejo de errores y feedback visual
- Almacenamiento de modelo entrenado
- Predicciones basadas en los últimos datos disponibles

### Mejoras futuras:

- Añadir gráficas interactivas de predicciones
- Implementar reentrenamiento desde la interfaz web
- Añadir panel de configuración del modelo
- Incluir historial de predicciones
- Mejorar la validación y visualización de errores

## Fase 4 — Interfaz web avanzada, evaluación y mejoras UX

En la Fase 4 hemos añadido una capa de interacción completa y artefactos de evaluación para facilitar el uso y la comprobación de la calidad del modelo.

Principales añadidos (issues resueltos):

- Interfaz web Flask integrada en `main.py` (rutas y plantillas): ahora la aplicación corre con `python main.py` y ofrece una página principal interactiva en `http://localhost:5000`.
- Predicción interactiva:
	- Botón rápido que usa la última secuencia de `outputs/X_test.npy` para producir una predicción.
	- Formulario de entrada manual: acepta secuencias coma-separadas y permite introducir tus propios valores para hacer predicciones en tiempo real.
	- Soporte para longitudes variables: las secuencias más cortas se rellenan (padding) con el último valor proporcionado y las más largas se truncan usando las últimas N observaciones.

- Evaluación y visualización:
	- Script `src/evaluate.py` que calcula predicciones sobre `X_test`, invierte la normalización con el scaler guardado y genera figuras y artefactos.
	- Se generan y guardan en `reports/figures/`:
		- `lstm_predictions.png` — predicciones vs reales
		- `lstm_error_series.png` — error por muestra (pred - real)
		- `lstm_error_hist.png` — histograma de errores
	- Los arrays originales en escala real se guardan en `data/results/` como `y_test_original.npy` y `y_pred_original.npy`.
	- La página web muestra estas figuras y una métrica MSE (si los archivos de evaluación existen).

- Mejora de experiencia de usuario (UI/UX):
	- Tema oscuro con tonos azul neón, tipografía tipo iOS/SF, y efectos sutiles de movimiento y glow (`static/style.css`).
	- Separación clara entre la sección de evaluación del dataset (gráficas y métricas) y la sección de interacción manual (formulario y resultado inmediato).
	- Animaciones y feedback visual cuando se genera una predicción manual (p. ej. pulso glow en el resultado).

Cómo ejecutar la Fase 4

1. Asegúrate de tener dependencias instaladas y el entorno activado:

```powershell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Genera los artefactos básicos si aún no existen:

```powershell
# Preprocesado (genera outputs/*.npy)
python main.py --preprocess-only

# Entrenamiento (genera models/lstm_model.h5)
python src/training.py

# Evaluación (genera figuras y data/results/*.npy)
python src/evaluate.py
```

3. Inicia la web e interactúa:

```powershell
python main.py
```

Visita: http://localhost:5000

Notas y recomendaciones:

- Si usas la entrada manual y no quieres preocuparte por la longitud exacta, la aplicación hará padding o truncado automáticamente; si prefieres otro comportamiento (padding con ceros, media, o rellenado por predicción recursiva), puedo cambiarlo.
- Para análisis más profundo, puedes generar el CSV con pares (y_true, y_pred) desde `data/results/` y visualizarlo con herramientas externas.
- Siguientes mejoras posibles: gráfica interactiva (Plotly), descarga CSV desde la UI, reentrenamiento desde la web con control de permisos, y logging/histórico de predicciones.

Estado: Fase 4 implementada (funcionalidad básica y UX). En progreso: pruebas de robustez, pequeños ajustes visuales y añadir descargas e interacción avanzada.

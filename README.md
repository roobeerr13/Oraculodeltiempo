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

Notas y mejoras futuras:

- Parametrizar `look_back` y la columna objetivo como argumentos CLI para experimentar sin editar código.
- Añadir tests unitarios que validen `create_dataset` con series sintéticas (happy path y casos con pocos datos).
- Implementar pipelines de entrenamiento y evaluación utilizando los arrays ya generados.


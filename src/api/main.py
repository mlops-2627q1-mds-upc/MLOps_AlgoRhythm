from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from loguru import logger

from src.config import MODELS_DIR

# ------------------------------------------------------------------------------
# 1. Variables globales y carga del modelo
# ------------------------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "random_forest.pkl"
model: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.
    Carga el modelo de 543MB en memoria al arrancar para evitar relecturas por petición.
    """
    global model
    logger.info(f"Cargando modelo pesado desde: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        logger.error(f"No se encontró el archivo del modelo en {MODEL_PATH}")
        raise FileNotFoundError(f"El archivo {MODEL_PATH} no existe. Ejecuta 'dvc pull' o entrena el modelo.")
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.success("Modelo cargado exitosamente en memoria.")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise e

    yield

    logger.info("Liberando recursos y apagando API...")
    model = None


# Inicialización de FastAPI
app = FastAPI(
    title="Spotify Song Popularity Prediction API",
    description="API para predecir la popularidad de una canción utilizando un modelo Random Forest.",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------------------
# 2. Esquema de entrada Pydantic (Un solo registro/canción)
# ------------------------------------------------------------------------------
class SongFeatures(BaseModel):
    """
    Esquema que define las variables de entrada requeridas para predecir 
    la popularidad de una sola canción.
    Ajusta los nombres de las columnas a las características exactas de tu dataset.
    """
    danceability: float = Field(..., example=0.65, description="Bailabilidad de la canción (0.0 a 1.0)")
    energy: float = Field(..., example=0.82, description="Energía de la canción (0.0 a 1.0)")
    key: int = Field(..., example=5, description="Tono de la canción (Pitch Class Notation)")
    loudness: float = Field(..., example=-4.2, description="Volumen general en dB")
    mode: int = Field(..., example=1, description="Modalidad (1 = Mayor, 0 = Menor)")
    speechiness: float = Field(..., example=0.05, description="Presencia de palabras habladas")
    acousticness: float = Field(..., example=0.12, description="Acústica de la canción")
    instrumentalness: float = Field(..., example=0.001, description="Probabilidad de que sea instrumental")
    liveness: float = Field(..., example=0.18, description="Presencia de audiencia en vivo")
    valence: float = Field(..., example=0.55, description="Positividad musical (0.0 a 1.0)")
    tempo: float = Field(..., example=122.0, description="Tempo estimado en BPM")
    duration_ms: int = Field(..., example=210000, description="Duración de la canción en milisegundos")
    explicit: int = Field(..., example=0, description="1 si tiene contenido explícito, 0 en caso contrario")
    genre: str = Field(..., example="pop", description="Género musical de la canción")
    
    # Si tu dataset ya fue featurizado a columnas One-Hot u omitió variables de texto,
    # asegúrate de incluir las columnas exactamente como las espera X_train.


class PredictionOutput(BaseModel):
    """Esquema de respuesta simple y directo."""
    predicted_popularity: float = Field(..., example=75.4, description="Popularidad predicha para la canción")


# ------------------------------------------------------------------------------
# 3. Endpoints de la API
# ------------------------------------------------------------------------------
@app.get("/health", tags=["Health Check"])
def health_check() -> Dict[str, str]:
    """Verifica si la API está arriba y el modelo cargado correctamente."""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está listo o falló al cargar.")
    return {"status": "ok", "model_loaded": "true"}


@app.post("/predict", response_model=PredictionOutput, tags=["Inferencia"])
def predict(features: SongFeatures) -> PredictionOutput:
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no se encuentra disponible.")

    try:
        data_dict = features.model_dump()
        genre_val = data_dict.pop("genre")  # Extraemos el género para el one-hot encoding

        # Inicializar un diccionario con ceros para todas las columnas que el Random Forest espera
        input_data = {col: 0 for col in model.feature_names_in_}

        # Asignar los valores numéricos enviados por el usuario
        for key, val in data_dict.items():
            if key in input_data:
                input_data[key] = val

        # Activar la columna one-hot correspondiente al género recibido
        genre_column = f"genre_{genre_val}"
        if genre_column in input_data:
            input_data[genre_column] = 1

        # Crear el DataFrame de una sola fila con el orden exacto de características
        input_df = pd.DataFrame([input_data])

        # Realizar la inferencia
        prediction = model.predict(input_df)

        return PredictionOutput(predicted_popularity=round(float(prediction[0]), 2))

    except Exception as e:
        logger.error(f"Error realizando la predicción: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Error en el formato de datos o en la inferencia: {str(e)}"
        )
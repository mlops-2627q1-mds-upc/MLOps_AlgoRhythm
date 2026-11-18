from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
import joblib
from pydantic import BaseModel, Field
from loguru import logger

from src.config import MODELS_DIR

from src.domain.user import User, UserRole
from src.database import get_db

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
    danceability: float = Field(..., json_schema_extra={"example": 0.65}, description="Bailabilidad de la canción (0.0 a 1.0)")
    energy: float = Field(..., json_schema_extra={"example": 0.82}, description="Energía de la canción (0.0 a 1.0)")
    key: int = Field(..., json_schema_extra={"example": 5}, description="Tono de la canción (Pitch Class Notation)")
    loudness: float = Field(..., json_schema_extra={"example": -4.2}, description="Volumen general en dB")
    mode: int = Field(..., json_schema_extra={"example": 1}, description="Modalidad (1 = Mayor, 0 = Menor)")
    speechiness: float = Field(..., json_schema_extra={"example": 0.05}, description="Presencia de palabras habladas")
    acousticness: float = Field(..., json_schema_extra={"example": 0.12}, description="Acústica de la canción")
    instrumentalness: float = Field(..., json_schema_extra={"example": 0.001}, description="Probabilidad de que sea instrumental")
    liveness: float = Field(..., json_schema_extra={"example": 0.18}, description="Presencia de audiencia en vivo")
    valence: float = Field(..., json_schema_extra={"example": 0.55}, description="Positividad musical (0.0 a 1.0)")
    tempo: float = Field(..., json_schema_extra={"example": 122.0}, description="Tempo estimado en BPM")
    duration_ms: int = Field(..., json_schema_extra={"example": 210000}, description="Duración de la canción en milisegundos")
    explicit: int = Field(..., json_schema_extra={"example": 0}, description="1 si tiene contenido explícito, 0 en caso contrario")
    genre: str = Field(..., json_schema_extra={"example": "pop"}, description="Género musical de la canción")
    
    # Si tu dataset ya fue featurizado a columnas One-Hot u omitió variables de texto,
    # asegúrate de incluir las columnas exactamente como las espera X_train.


class PredictionOutput(BaseModel):
    """Esquema de respuesta simple y directo."""
    predicted_popularity: float = Field(..., json_schema_extra={"example": 75.4}, description="Popularidad predicha para la canción")



# -------------- TEST AUTHORIZATION --------------

def get_current_user(db: Session = Depends(get_db)) -> User:
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    return current_user

def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403, 
            detail="You do not have permission to access this resource."
        )
    return current_user

@app.get("/public-endpoint")
def function_public_endpoint():
    return {"message": "Welcome! Anyone can access this endpoint."}

@app.get("/signed-in-endpoint")
def function_signed_endpoint(current_user: User = Depends(get_current_active_user)):
    return {
        "message": f"Hello {current_user.username}! You have access to this signed endpoint.",
        "user_role": current_user.role
    }

@app.get("/users/{user_id}/owner-endpoint")
def function_owner_endpoint(user_id: int, current_user: User = Depends(get_current_active_user)):
    if current_user.user_id != user_id:
        raise HTTPException(
            status_code=403, 
            detail="You do not have permission to access this resource."
            )
    result = "You are the owner"
    return {"message": f"Hello {current_user.username} {result} of this resource." }
    
@app.get("/admin-endpoint")
def function_admin_endpoint(current_user: User = Depends(get_admin_user)):
    return {
        "message": f"Hello {current_user.username}! You have access to this admin endpoint.",
        "user_role": current_user.role
    }

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
    return None
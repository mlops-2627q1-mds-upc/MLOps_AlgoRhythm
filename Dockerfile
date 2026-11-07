FROM python:3.10-slim

WORKDIR /code

# 1. Copiar e instalar únicamente dependencias de producción
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar únicamente los artefactos del modelo y el código de la API
COPY models/ ./models/
COPY src/ ./src/

# 3. Exponer el puerto 8080 (requerido para el mapeo NAT de Virtech)
EXPOSE 8080

# 4. Iniciar Uvicorn apuntando a src/api/main.py
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
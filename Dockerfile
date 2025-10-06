# Usa una imagen ligera de Python 3.11
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y luego instalarlos
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Exponer puerto 8080 (Cloud Run usa 8080)
EXPOSE 8080

# Comando de arranque con uvicorn
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--loop", "asyncio"]

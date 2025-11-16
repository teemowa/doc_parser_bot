FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем локальные модели (fallback)
COPY ./local_models /app/local_models

# Копируем основной код
COPY main.py .

# Модели с HuggingFace будут автоматически скачаны при первом запуске
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
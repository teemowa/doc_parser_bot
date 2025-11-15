FROM python:3.10-slim

WORKDIR /app

# Копируем код stamp2vec в образ
COPY ./stamp2vec /app/stamp2vec

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем *также* зависимости из самого stamp2vec
# (Это самый безопасный способ)
RUN pip install --no-cache-dir -r /app/stamp2vec/requirements.txt

COPY main.py .

# Модели будут скачаны и кэшированы при первом запуске
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from ultralytics import YOLO
from qrdet import QRDetector
import cv2
import numpy as np
import io
import logging
import sys
from PIL import Image 

# --- Настройка для stamp2vec ---
sys.path.append('/app/stamp2vec')
from pipelines.detection.yolo_stamp import YoloStampPipeline 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загружаем ВСЕ 4 модели при старте
    logger.info("Загрузка всех ML моделей (из интернета)...")
    
    # --- Эти 2 модели грузим локально ---
    models['signature'] = YOLO("/app/local_models/signature_model.pt")
    models['table'] = YOLO("/app/local_models/table_model.pt")
    
    models['qr'] = QRDetector() # Эта модель встроена в pip-пакет
    
    # --- Эту 1 модель грузим из интернета ---
    try:
        models['stamp'] = YoloStampPipeline.from_pretrained('stamps-labs/yolo-stamp')
        logger.info("Модель stamp2vec (YoloStampPipeline) загружена.")
    except Exception as e:
        logger.error(f"!!! ОШИБКА ЗАГРУЗКИ STAMP2VEC: {e}")

    logger.info("Все ML модели загружены.")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/detect_all")
async def detect_all_objects(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение.")

        # --- 1 & 2: Модели YOLO (Подписи, Таблицы) ---
        sig_results = models['signature'].predict(img_bgr, verbose=False)
        signatures = sig_results[0].boxes.xyxy.tolist()

        table_results = models['table'].predict(img_bgr, verbose=False)
        tables = table_results[0].boxes.xyxy.tolist()

        # --- 3: Модель QR-кодов ---
        qr_results = models['qr'].detect(img_bgr, is_bgr=True)
        # --- ИСПРАВЛЕНО: 'bbox' -> 'bbox_xyxy' и добавлена проверка ---
        qr_codes = []
        if qr_results:
            qr_codes = [d['bbox_xyxy'] for d in qr_results if d.get('bbox_xyxy')]

        # --- 4: Модель Печатей (stamp2vec) ---
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # --- ИСПРАВЛЕНО: Передаем PIL Image ---
        img_pil = Image.fromarray(img_rgb)
        
        stamp_boxes_tensor = models['stamp'](image=img_pil) 
        stamps = stamp_boxes_tensor.cpu().numpy().tolist()

        return {
            "signatures": signatures,
            "tables": tables,
            "qr_codes": qr_codes,
            "stamps": stamps
        }

    except Exception as e:
        logger.error(f"Ошибка в ml_service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
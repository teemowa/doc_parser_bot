from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from ultralytics import YOLO
import cv2
import numpy as np
import io
import logging
import os
import sys
from PIL import Image
from huggingface_hub import hf_hub_download
from qrdet import QRDetector
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

SIGNATURE_CONF = float(os.getenv("SIGNATURE_CONF", "0.15"))
TABLE_CONF = float(os.getenv("TABLE_CONF", "0.15"))
QR_CONF = float(os.getenv("QR_CONF", "0.3"))  # Понижен для детекции мелких QR
STAMP_CIRCULAR_CONF = float(os.getenv("STAMP_CIRCULAR_CONF", "0.20"))
STAMP_RECTANGULAR_CONF = float(os.getenv("STAMP_RECTANGULAR_CONF", "0.20"))
STAMP_TABLE_IOU_THRESHOLD = float(os.getenv("STAMP_TABLE_IOU_THRESHOLD", "0.3"))
# Коэффициент увеличения разрешения для лучшей детекции QR (2.5 = 2.5x увеличение)
QR_UPSCALE_FACTOR = float(os.getenv("QR_UPSCALE_FACTOR", "2.5"))
# Применять ли предобработку изображения (контраст, резкость)
QR_ENHANCE_IMAGE = os.getenv("QR_ENHANCE_IMAGE", "true").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
models = {}


def upscale_image_for_qr(img_bgr, scale_factor=2.5, enhance=True):
    """
    Увеличивает разрешение изображения для лучшей детекции QR кодов.
    Использует INTER_CUBIC для качественного увеличения.
    Опционально применяет предобработку для улучшения контраста и резкости.
    """
    height, width = img_bgr.shape[:2]
    
    # Адаптивное масштабирование: для маленьких изображений увеличиваем больше
    adaptive_scale = scale_factor
    if width < 1500 or height < 1500:
        adaptive_scale = scale_factor * 1.2  # +20% для маленьких изображений
        logger.info(f"Применено адаптивное масштабирование: {scale_factor} -> {adaptive_scale:.2f}")
    
    if adaptive_scale == 1.0:
        return img_bgr, adaptive_scale
    
    new_width = int(width * adaptive_scale)
    new_height = int(height * adaptive_scale)
    
    # Увеличение с высоким качеством
    upscaled = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Предобработка для улучшения детекции
    if enhance:
        # 1. Мягкое увеличение резкости (помогает с границами QR без артефактов)
        # Используем Unsharp Mask для более качественного результата
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        upscaled = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
        
        # 2. Адаптивное улучшение контраста (CLAHE) с оптимизированными параметрами
        # Конвертируем в LAB, применяем CLAHE к L-каналу
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # clipLimit=2.5 для чуть более агрессивного контраста
        # tileGridSize=(8,8) для локального улучшения
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        upscaled = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        logger.info(f"Применена предобработка: Unsharp Mask + CLAHE")
    
    logger.info(f"Изображение увеличено для QR детекции: {width}x{height} -> {new_width}x{new_height}")
    return upscaled, adaptive_scale


def scale_detections_back(detections, scale_factor, img_shape=None, expand_small=True):
    """
    Масштабирует координаты детекций обратно к оригинальному размеру.
    Опционально расширяет границы для мелких QR кодов.
    """
    if scale_factor == 1.0 and not expand_small:
        return detections
    
    scaled_detections = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox_xyxy']
        
        # Масштабируем обратно
        x1_scaled = x1 / scale_factor
        y1_scaled = y1 / scale_factor
        x2_scaled = x2 / scale_factor
        y2_scaled = y2 / scale_factor
        
        # Для мелких QR немного расширяем границы (на 2-3 пикселя)
        if expand_small:
            width = x2_scaled - x1_scaled
            height = y2_scaled - y1_scaled
            area = width * height
            
            # Если QR маленький (площадь < 1500 пикселей), расширяем границы
            if area < 1500:
                expand_px = 3  # Расширение на 3 пикселя с каждой стороны
                x1_scaled = max(0, x1_scaled - expand_px)
                y1_scaled = max(0, y1_scaled - expand_px)
                x2_scaled = x2_scaled + expand_px
                y2_scaled = y2_scaled + expand_px
                
                # Ограничиваем границами изображения
                if img_shape is not None:
                    h, w = img_shape[:2]
                    x2_scaled = min(w, x2_scaled)
                    y2_scaled = min(h, y2_scaled)
        
        scaled_detections.append([
            int(x1_scaled),
            int(y1_scaled),
            int(x2_scaled),
            int(y2_scaled)
        ])
    return scaled_detections


def compute_iou(box1, box2):
    """
    Вычисляет IoU (Intersection over Union) между двумя боксами.
    box1, box2: [x1, y1, x2, y2]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Координаты пересечения
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    # Если нет пересечения
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    # Площади
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def download_hf_model(repo_id, filename="best.pt"):
    """Скачивает модель с HuggingFace Hub"""
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info(f"✓ Модель скачана: {repo_id}")
        return model_path
    except Exception as e:
        logger.warning(f"✗ Не удалось скачать {repo_id}: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Загрузка всех ML моделей...")
    
    # 1. Подписи - специализированная модель tech4humans/yolov8s-signature-detector
    try:
        sig_path = download_hf_model("tech4humans/yolov8s-signature-detector")
        if sig_path:
            models['signature'] = YOLO(sig_path)
            logger.info("✓ Модель подписей загружена (tech4humans/yolov8s-signature-detector)")
        else:
            raise Exception("HF download failed")
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки модели подписей: {e}")
        logger.info("Пробую загрузить локальную модель подписей...")
        try:
            if os.path.exists("/app/local_models/signature_model.pt"):
                models['signature'] = YOLO("/app/local_models/signature_model.pt")
                logger.info("✓ Локальная модель подписей загружена")
            else:
                models['signature'] = YOLO("yolov8m.pt")
                logger.warning("⚠ Использую YOLOv8m для подписей (может работать хуже)")
        except Exception as e2:
            models['signature'] = YOLO("yolov8m.pt")
    
    # 2. Таблицы - ОТКЛЮЧЕНО
    models['table'] = None
    logger.info("⚠ Детекция таблиц отключена")
    
    # 3. QR-коды - специализированная библиотека qrdet (YOLOv8 based)
    try:
        # model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
        # Используем 'l' для максимальной точности и детекции мелких QR
        qr_model_size = os.getenv("QR_MODEL_SIZE", "l")
        # nms_iou=0.4 - более строгий NMS для лучших границ
        models['qr'] = QRDetector(model_size=qr_model_size, conf_th=QR_CONF, nms_iou=0.4)
        logger.info(f"✓ QR детектор загружен (qrdet с моделью '{qr_model_size}', nms_iou=0.4)")
    except Exception as e:
        logger.error(f"Критическая ошибка загрузки qrdet: {e}")
        # Fallback на стандартную YOLO (не рекомендуется для QR)
        models['qr'] = YOLO("yolov8m.pt")
        logger.warning("⚠ Использую YOLOv8m для QR (может работать хуже)")
    
    # 4. Печати - модель Ooredoo-Group/ooredoo-stamp-detection (transformers)
    try:
        models['stamp_processor'] = AutoImageProcessor.from_pretrained("Ooredoo-Group/ooredoo-stamp-detection")
        models['stamp_model'] = AutoModelForObjectDetection.from_pretrained("Ooredoo-Group/ooredoo-stamp-detection")
        logger.info("✓ Модель печатей загружена (Ooredoo-Group/ooredoo-stamp-detection)")
    except Exception as e:
        logger.error(f"✗ Критическая ошибка загрузки модели печатей: {e}")
        logger.warning("⚠ Fallback: использую YOLOv8m для печатей")
        models['stamp_processor'] = None
        models['stamp_model'] = None
        models['stamp_fallback'] = YOLO("yolov8m.pt")

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

        # --- 1: Модель подписей ---
        sig_results = models['signature'].predict(img_bgr, verbose=False, conf=SIGNATURE_CONF)
        signatures = sig_results[0].boxes.xyxy.tolist()

        # --- 2: Модель таблиц (ОТКЛЮЧЕНО) ---
        tables = []

        # --- 3: Модель QR-кодов (qrdet с увеличением разрешения и предобработкой) ---
        # Увеличиваем изображение для лучшей детекции мелких QR кодов
        img_upscaled, scale_factor = upscale_image_for_qr(
            img_bgr, 
            scale_factor=QR_UPSCALE_FACTOR, 
            enhance=QR_ENHANCE_IMAGE
        )
        
        # Детекция на увеличенном изображении
        if isinstance(models['qr'], QRDetector):
            # Используем qrdet
            qr_detections = models['qr'].detect(image=img_upscaled, is_bgr=True)
            # Масштабируем координаты обратно с расширением границ для мелких QR
            qr_codes = scale_detections_back(
                qr_detections, 
                scale_factor, 
                img_shape=img_bgr.shape,
                expand_small=True
            )
            logger.info(f"qrdet нашёл {len(qr_codes)} QR кодов")
        else:
            # Fallback на YOLO
            qr_results = models['qr'].predict(img_upscaled, verbose=False, conf=QR_CONF)
            qr_boxes = qr_results[0].boxes.xyxy.tolist()
            # Масштабируем координаты обратно
            qr_codes = [[int(x1/scale_factor), int(y1/scale_factor), 
                        int(x2/scale_factor), int(y2/scale_factor)] 
                       for x1, y1, x2, y2 in qr_boxes]

        # --- 4: Модель печатей (Ooredoo transformers) ---
        stamps_raw = []
        
        if models.get('stamp_processor') and models.get('stamp_model'):
            # Конвертируем BGR в RGB для transformers
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            # Inference
            inputs = models['stamp_processor'](images=pil_image, return_tensors="pt")
            outputs = models['stamp_model'](**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = models['stamp_processor'].post_process_object_detection(
                outputs, 
                target_sizes=target_sizes,
                threshold=STAMP_CIRCULAR_CONF  # Используем как общий порог
            )[0]
            
            # Конвертируем в формат [x1, y1, x2, y2]
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box_list = box.tolist()
                stamps_raw.append([
                    int(box_list[0]),  # x1
                    int(box_list[1]),  # y1
                    int(box_list[2]),  # x2
                    int(box_list[3])   # y2
                ])
            logger.info(f"Ooredoo модель нашла {len(stamps_raw)} печатей")
        
        elif models.get('stamp_fallback'):
            # Fallback на YOLO
            stamp_results = models['stamp_fallback'].predict(img_bgr, verbose=False, conf=STAMP_CIRCULAR_CONF)
            stamps_raw = stamp_results[0].boxes.xyxy.tolist()
            logger.info(f"Fallback YOLO нашла {len(stamps_raw)} печатей")
        
        # Фильтрация отключена (таблицы не детектируются)
        stamps = stamps_raw

        logger.info(
            "Обнаружено объектов: подписи=%d, таблицы=%d, QR=%d, печати=%d",
            len(signatures), len(tables), len(qr_codes), len(stamps)
        )

        return {
            "signatures": signatures,
            "tables": tables,
            "qr_codes": qr_codes,
            "stamps": stamps
        }

    except Exception as e:
        logger.error(f"Ошибка в ml_service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

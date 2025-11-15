import logging
import os
import sys
import asyncio
import json
from io import BytesIO

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import fitz  # PyMuPDF
import cv2
import numpy as np

# --- Настройка ---
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- ОСТАЛСЯ ТОЛЬКО ОДИН СЕРВИС ---
ML_SERVICE_URL = "http://ml-service:8000/detect_all" # <--- Изменено

# (Словари PDF_COLORS и CV_COLORS остаются БЕЗ ИЗМЕНЕНИЙ)
PDF_COLORS = {
    "signature": (0, 0, 1), "qr_code": (0, 1, 0), "stamp": (1, 0, 0), "table": (1, 0.5, 0),
}
CV_COLORS = {
    "signature": (255, 0, 0), "qr_code": (0, 255, 0), "stamp": (0, 0, 255), "table": (0, 128, 255),
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (Без изменений) ...
    await update.message.reply_text(
        "Привет! Отправь мне PDF-документ. Я найду на нем:\n"
        "- Подписи\n- Таблицы\n- QR-коды\n- Печати\n\n"
        "И пришлю в ответ PDF-отчет и тепловую карту."
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (Без изменений) ...
    file_name = update.message.document.file_name
    if not file_name.lower().endswith('.pdf'):
        await update.message.reply_text("Пожалуйста, прикрепите документ в формате PDF.")
        return
    await update.message.reply_text(f"Получил ваш PDF: {file_name}\nНачинаю обработку... Это может занять несколько минут.")
    try:
        pdf_file = await context.bot.get_file(update.message.document.file_id)
        pdf_bytes = await pdf_file.download_as_bytearray()
        results = await process_pdf_and_generate_reports(pdf_bytes, file_name, update)
        
        if results["annotated_pdf_bytes"]:
            annotated_pdf_file = BytesIO(results["annotated_pdf_bytes"])
            annotated_pdf_file.name = f"{os.path.splitext(file_name)[0]}_ANNOTATED.pdf"
            await context.bot.send_document(chat_id=update.effective_chat.id, document=annotated_pdf_file, caption="Идея 2: Ваш PDF-документ с пометками.")
        
        if results["heatmap_image_bytes"]:
            heatmap_file = BytesIO(results["heatmap_image_bytes"])
            heatmap_file.name = f"{os.path.splitext(file_name)[0]}_HEATMAP.jpg"
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=heatmap_file, caption="Идея 3: Тепловая карта найденных объектов.")

        final_json_str = json.dumps(results["final_json"], indent=2, ensure_ascii=False)
        if len(final_json_str) > 4096:
            await update.message.reply_text("JSON с результатами (для отладки):", quote=False)
            json_file = BytesIO(final_json_str.encode('utf-8'))
            json_file.name = f"{os.path.splitext(file_name)[0]}_results.json"
            await context.bot.send_document(chat_id=update.effective_chat.id, document=json_file)
        else:
            await update.message.reply_text(f"```json\n{final_json_str}\n```", parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Ошибка обработки PDF: {e}", exc_info=True)
        await update.message.reply_text(f"Произошла ошибка во время анализа документа: {e}")


async def process_pdf_and_generate_reports(pdf_bytes, file_name, update: Update):
    final_json = {file_name: {}}
    all_annotations_for_heatmap = []
    page_dimensions_for_heatmap = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_key = f"page_{page_num + 1}"
        logger.info(f"Обработка страницы: {page_key}...")
        
        img_dpi = 200
        pix = page.get_pixmap(dpi=img_dpi)
        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height
        page_dimensions_for_heatmap.append((pix.width, pix.height))

        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        is_success, buffer = cv2.imencode(".jpg", img_bgr)
        if not is_success: continue
        image_bytes = buffer.tobytes()
        files = {'file': (f"{page_key}.jpg", image_bytes, 'image/jpeg')}

        page_size_json = {"width": pix.width, "height": pix.height}
        annotations_json = []
        page_annotations_for_drawing = []

        # --- БЛОК ЗАПРОСА СТАЛ НАМНОГО ПРОЩЕ ---
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                # --- УБРАЛИ ASYNCIO.GATHER, ТЕПЕРЬ ТУТ ОДИН ЗАПРОС ---
                response = await client.post(ML_SERVICE_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    for coords in data.get('signatures', []):
                        page_annotations_for_drawing.append(("signature", coords))
                    for coords in data.get('qr_codes', []):
                        page_annotations_for_drawing.append(("qr_code", coords))
                    for coords in data.get('tables', []):
                        page_annotations_for_drawing.append(("table", coords))
                    for coords in data.get('stamps', []): # <-- ДОБАВЛЕНО
                        page_annotations_for_drawing.append(("stamp", coords)) # <-- ДОБАВЛЕНО
                else:
                    logger.error(f"ML-сервис вернул ошибку: {response.status_code} - {response.text}")

                # --- ОБРАБОТКА (БЕЗ ИЗМЕНЕНИЙ) ---
                for category, coords_xyxy in page_annotations_for_drawing:
                    bbox = convert_xyxy_to_bbox(coords_xyxy)
                    annotations_json.append({"category": category, "bbox": bbox})
                    asyncio.create_task(send_crop(img_bgr, bbox, category, page_key, update))
                    pdf_rect = fitz.Rect(bbox["x"] * scale_x, bbox["y"] * scale_y, (bbox["x"] + bbox["width"]) * scale_x, (bbox["y"] + bbox["height"]) * scale_y)
                    page.draw_rect(pdf_rect, color=PDF_COLORS.get(category, (0,0,0)), width=1.5)
                    all_annotations_for_heatmap.append((page_num, category, bbox))

            except Exception as e:
                logger.error(f"Ошибка при запросе к ML-сервису: {e}", exc_info=True)

        final_json[file_name][page_key] = {"page_size": page_size_json, "annotations": annotations_json}
    
    # --- Генерация отчетов (БЕЗ ИЗМЕНЕНИЙ) ---
    # --- ПРАВИЛЬНО ---
    annotated_pdf_bytes = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    heatmap_bytes = generate_heatmap(all_annotations_for_heatmap, page_dimensions_for_heatmap)
    
    return {
        "final_json": final_json,
        "annotated_pdf_bytes": annotated_pdf_bytes,
        "heatmap_image_bytes": heatmap_bytes
    }

def generate_heatmap(annotations, page_dimensions):
    # ... (Эта функция остается БЕЗ ИЗМЕНЕНИЙ) ...
    if not page_dimensions: return None
    try:
        max_width = max(w for w, h in page_dimensions)
        total_height = sum(h for w, h in page_dimensions)
        heatmap_img = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
        overlay = heatmap_img.copy()
        page_y_offset = 0
        for page_num, (page_w, page_h) in enumerate(page_dimensions):
            page_annotations = [a for a in annotations if a[0] == page_num]
            for _, category, bbox in page_annotations:
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                x1, y1 = x, y + page_y_offset
                x2, y2 = x + w, y + h + page_y_offset
                color = CV_COLORS.get(category, (128, 128, 128))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
            page_y_offset += page_h
        alpha = 0.3
        final_heatmap = cv2.addWeighted(overlay, alpha, heatmap_img, 1 - alpha, 0)
        is_success, buffer = cv2.imencode(".jpg", final_heatmap)
        return buffer.tobytes() if is_success else None
    except Exception as e:
        logger.error(f"Ошибка генерации тепловой карты: {e}", exc_info=True)
        return None

def convert_xyxy_to_bbox(coords):
    # ... (Эта функция остается БЕZ ИЗМЕНЕНИЙ) ...
    # Учтем, что stamp2vec может вернуть [x1, y1, x2, y2] как float
    x1, y1, x2, y2 = map(int, coords) 
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

async def send_crop(image_np, bbox, category_name, page_key, update: Update):
    # ... (Эта функция остается БЕЗ ИЗМЕНЕНИЙ) ...
    try:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        y_max, x_max = image_np.shape[:2]
        crop_img = image_np[max(0, y):min(y_max, y + h), max(0, x):min(x_max, x + w)]
        is_success, buffer = cv2.imencode(".jpg", crop_img)
        if not is_success: return
        photo_bytes = BytesIO(buffer.tobytes())
        await update.message.reply_photo(photo=photo_bytes, caption=f"Найдено: {category_name} (страница: {page_key})")
    except Exception as e:
        logger.error(f"Ошибка отправки кропа: {e}")

def main():
    # ... (Эта функция остается БЕЗ ИЗМЕНЕНИЙ) ...
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TOKEN:
        logger.critical("Токен TELEGRAM_TOKEN не найден в .env файле!")
        sys.exit("Токен не найден.")
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    logger.info("Бот запускается (через Docker)...")
    application.run_polling()

if __name__ == "__main__":
    main()
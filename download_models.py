from huggingface_hub import login, hf_hub_download
import os

# 1. Логинимся. Он попросит токен прямо здесь, в терминале.
login()

# 2. Определяем, куда сохранить
save_dir = "ml_service/local_models"
file_name = "signature_model.pt"
full_path = os.path.join(save_dir, file_name)

# 3. Скачиваем "проблемную" модель
print(f"Downloading signature model to {full_path}...")
hf_hub_download(
    repo_id="tech4humans/yolov8s-signature-detector",
    filename="yolov8s.pt",  # <--- ИСПРАВЛЕНО
    local_dir=save_dir,
    local_dir_use_symlinks=False
)

# 4. Переименовываем (скачается как yolov8s.pt)
os.rename(os.path.join(save_dir, "yolov8s.pt"), full_path) # <--- ИСПРАВЛЕНО
print(f"Successfully downloaded and renamed to {file_name}")

# 5. Скачиваем "легкую" модель
print("Downloading table model...")
os.system("curl -L -o ml_service/local_models/table_model.pt https://huggingface.co/keremberke/yolov8m-table-extraction/resolve/main/best.pt")

print("All models downloaded successfully.")
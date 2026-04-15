
import os
import requests
from src.utils.logger import log

def download_model(url, dest_path):
    if os.path.exists(dest_path):
        log.info(f"Model already exists at {dest_path}")
        return

    log.info(f"Downloading model from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info(f"Model saved to {dest_path}")
    except Exception as e:
        log.error(f"Failed to download model: {e}")

if __name__ == "__main__":
    DEST_DIR = "models"
    os.makedirs(DEST_DIR, exist_ok=True)
    DEST_PATH = os.path.join(DEST_DIR, "yolov8n_plate.pt")

    # List of candidate URLs
    URLS = [
        # Keremberke v0.1.3 (Likely valid)
        "https://github.com/keremberke/yolov8-license-plate-recognition/releases/download/v0.1.3/license_plate_yolov8n.pt",
        # Hugging Face mirror
        "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/license_plate_yolov8n.pt",
        # Fallback: Standard YOLOv8n (Will detect 'car', 'truck' etc, but allows pipeline verification)
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    ]

    for url in URLS:
        try:
            download_model(url, DEST_PATH)
            if os.path.exists(DEST_PATH) and os.path.getsize(DEST_PATH) > 0:
                log.info(f"Successfully downloaded model from {url}")
                break
        except Exception as e:
            log.warning(f"Failed to download from {url}: {e}")
    
    if not os.path.exists(DEST_PATH):
        log.error("All download attempts failed. Please manually place 'yolov8n_plate.pt' in 'models/' folder.")

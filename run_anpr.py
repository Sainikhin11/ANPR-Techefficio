import setuptools
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import yaml
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.utils.logger import log, setup_logger
from src.utils.gpu import check_gpu, log_system_stats
from src.utils.anpr_postprocess import is_similar_plate
from src.core.anpr_controller import ANPRController

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def add_recent_plate(recent_plates, plate, conf):
    for i, (existing_plate, existing_conf) in enumerate(recent_plates):
        if is_similar_plate(existing_plate, plate):
            if conf > existing_conf:
                recent_plates[i] = (plate, conf)
            return
    recent_plates.insert(0, (plate, conf))

def process_raw_images(controller, input_dir="data/raw_images", output_dir="data/raw_images_results"):
    import glob
    os.makedirs(output_dir, exist_ok=True)
    images = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
             glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))
             
    log.info(f"Processing {len(images)} raw images from {input_dir}...")
    for img_path in images:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        detections = controller.detector.detect(img)
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            bh, bw = y2 - y1, x2 - x1
            if bh > 0 and (bw / bh) < 1.5:
                continue # ignore two wheelers
                
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            px1, py1 = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
            px2, py2 = min(img.shape[1], int(x2) + pad_x), min(img.shape[0], int(y2) + pad_y)
            
            crop = img[py1:py2, px1:px2]
            if crop.size == 0: continue
            
            text, ocr_conf = controller.ocr.recognize(crop)
            if text and ocr_conf > 0.4:
                _, text = controller.ocr.validate(text)
                if len(text) >= 4:
                    label = f"[{text}] {ocr_conf:.2f} LOCKED"
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(img, label, (int(x1), int(max(0, y1-10))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
        cv2.imwrite(os.path.join(output_dir, filename), img)
    log.info(f"Raw images saved to {output_dir}")

def main():
    log.info("Starting ANPR System (Real-time Display Mode)...")
    
    # 1. Config & Setup
    config = load_config()
    setup_logger()
    check_gpu()
    
    # 2. Initialize Controller
    try:
        controller = ANPRController(config)
        log.info("ANPR Controller Initialized.")
        
        # 2.5 Run on raw images as requested, before starting video
        # process_raw_images(controller)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        log.error(f"Failed to init Controller: {e}")
        return

    # 3. Video Capture (Direct)
    source = config['camera']['source']
    log.info(f"Opening video source: {source}")
    
    # Handle webcam index vs file path
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        log.error(f"Failed to open video source: {source}")
        print(f"Error: Could not open video source: {source}")
        return

    # Create Window explicitly
    window_name = "ANPR Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # 4. Main Loop
    prev_time = time.time()
    frame_count = 0
    fps = 0
    
    # Accumulator for recent plates display
    recent_plates = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of video stream. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # --- Process Frame ---
            display_frame, events = controller.process_frame(frame)
            
            h, w = display_frame.shape[:2]
            # Update recent plates
            if events:
                for ev in events:
                    add_recent_plate(recent_plates, ev['plate_number'], ev['confidence'])
                
                # Keep only last 5
                if len(recent_plates) > 5:
                    recent_plates = recent_plates[:5]

            # Draw Sidebar for extracted numbers
            if recent_plates:
                # Semi-transparent background
                overlay = display_frame.copy()
                box_height = 50 + (len(recent_plates) * 35)
                cv2.rectangle(overlay, (w - 350, 10), (w - 20, box_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
                
                # Title
                cv2.putText(display_frame, "Extracted Plates", (w - 330, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # List items
                for i, (plate, conf) in enumerate(recent_plates):
                    text = f"{plate} ({conf:.2f})"
                    cv2.putText(display_frame, text, (w - 330, 80 + i * 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show Frame
            cv2.imshow(window_name, display_frame)
            
            # WaitKey (Essential for display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("Quit key pressed.")
                break
                
    except KeyboardInterrupt:
        log.info("Stopping...")
    except Exception as e:
        log.error(f"Runtime Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("System Stopped.")

if __name__ == "__main__":
    main()

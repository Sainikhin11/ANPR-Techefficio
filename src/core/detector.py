import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from src.utils.logger import log

torch.set_num_threads(1)
cv2.setNumThreads(1)

class PlateDetector:
    def __init__(self, model_path, tracker_config="bytetrack.yaml", conf_thresh=0.45, iou_thresh=0.45, img_size=640, device=None):
        """
        Initializes the YOLOv8 Detector.
        """
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" and torch.cuda.is_available() else ""
        slow_fp16_gpu = "GTX 16" in gpu_name
        self.use_half = self.device == "cuda" and torch.cuda.is_available() and not slow_fp16_gpu
        
        try:
            log.info(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(model_path)
            self.model.fuse()
            
            # Move to device and enable FP16 after fusion to avoid Conv+BN dtype mismatches.
            self.model.to(self.device)
            if self.use_half and getattr(self.model, 'model', None):
                self.model.model.half()
            log.info(f"Model moved to {self.device} (half={self.use_half})")
            
            # Warmup
            log.info("Warming up model...")
            dummy_input = torch.zeros(1, 3, img_size, img_size, device=self.device)
            if self.use_half:
                dummy_input = dummy_input.half()
            self.model(dummy_input, verbose=False)
            log.info("Model warmup complete.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            log.error(f"Failed to load detection model: {e}")
            raise e

    def detect(self, img):
        """
        Runs object detection without tracking.
        Args:
            img (np.array): Input image.
        Returns:
            list: List of detections [[x1, y1, x2, y2, conf, cls_id], ...].
        """
        try:
            results = self.model.predict(
                img, 
                imgsz=self.img_size, 
                conf=self.conf_thresh, 
                iou=self.iou_thresh,
                verbose=False,
                device=self.device,
                half=self.use_half
            )[0]
            
            detections = []
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confs, clss):
                    # Filter for class 0 (license_plate) only
                    if int(cls_id) != 0:
                        continue
                        
                    x1, y1, x2, y2 = box
                    detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id)])
            
            return detections
            
        except Exception as e:
            log.error(f"Detection failed: {e}")
            return []

    def detect_and_track(self, check_frame):
        """
        Runs inference and tracking on a single frame.
        Args:
            check_frame (np.array): Original frame (BGR).
        Returns:
            list: List of tracks [[x1, y1, x2, y2, track_id, conf, cls_id], ...] scaled to original size.
        """
        try:
            # Inference with tracking
            results = self.model.track(
                check_frame, 
                imgsz=self.img_size, 
                conf=self.conf_thresh, 
                iou=self.iou_thresh, 
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
                device=self.device,
                half=self.use_half
            )[0]
            
            tracks = []
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy()
                
                for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, clss):
                    x1, y1, x2, y2 = box
                    tracks.append([int(x1), int(y1), int(x2), int(y2), int(track_id), float(conf), int(cls_id)])
            
            return tracks

        except Exception as e:
            log.error(f"Tracking failed: {e}")
            return []

        except Exception as e:
            log.error(f"Inference failed: {e}")
            return []

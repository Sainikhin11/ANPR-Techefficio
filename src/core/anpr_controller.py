import cv2
import time
import numpy as np
import Levenshtein
import re
import os
from collections import defaultdict, deque
from src.core.detector import PlateDetector
from src.core.ocr import OCRProcessor
from src.utils.logger import log
from src.utils.anpr_postprocess import correct_anpr_format, is_blurry, final_voting, adaptive_conf_filter, is_fast_motion, is_similar_plate

def clean_plate(text):
    if text is None:
        return ""
    text = str(text).upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def correct_plate(text):
    if len(text) < 8 or len(text) > 10:
        return None

    num_to_alpha = {'0':'O','1':'I','2':'Z','5':'S','8':'B'}
    alpha_to_num = {'O':'0','I':'1','Z':'2','S':'5','B':'8'}

    text = list(text)

    # State code (letters)
    for i in [0,1]:
        if i < len(text) and text[i] in num_to_alpha:
            text[i] = num_to_alpha[text[i]]

    # District code (numbers)
    for i in [2,3]:
        if i < len(text) and text[i] in alpha_to_num:
            text[i] = alpha_to_num[text[i]]

    # Last 4 always numbers
    for i in range(len(text)-4, len(text)):
        if i >= 0 and i < len(text) and text[i] in alpha_to_num:
            text[i] = alpha_to_num[text[i]]

    return "".join(text)

class VehicleTrack:
    def __init__(self, track_id, timestamp):
        self.track_id = track_id
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.status = "reading" # reading -> locked
        
        self.ocr_buffer = deque(maxlen=5) # Stores (text, conf)
        
        self.final_text = None
        self.final_conf = 0.0
        self.best_crop = None
        
        self.best_height = 0
        self.best_frame = None
        
        self.current_text = None
        self.current_conf = 0.0
        
        # Debug Metrics
        self.total_frames_alive = 0
        
    def add_ocr(self, text, conf, crop=None):
        if self.status == "locked":
            return
            
        self.ocr_buffer.append((text, conf))
        
        self.current_text = text
        self.current_conf = conf
        
        if crop is not None:
             # Keep crop with highest confidence or just latest if sufficient
             if self.best_crop is None or conf > self.final_conf:
                 self.best_crop = crop
                 
        log.debug(f"Track {self.track_id} buffer: {[t for t, c in self.ocr_buffer]}")

    def check_stability(self, stability_bound=3):
        if self.status == "locked":
            return True
            
        if len(self.ocr_buffer) < stability_bound:
            return False

        # Build fuzzy clusters
        clusters = [] # list of lists of (text, conf)
        
        for text, conf in self.ocr_buffer:
            added = False
            for cluster in clusters:
                proto = cluster[0][0]
                max_len = max(len(text), len(proto))
                if max_len == 0:
                    continue
                dist = Levenshtein.distance(text, proto)
                similarity = 1 - (dist / max_len)
                
                if similarity >= 0.8:
                    cluster.append((text, conf))
                    added = True
                    break
            
            if not added:
                clusters.append([(text, conf)])
                
        # Check for lock condition
        for cluster in clusters:
            if len(cluster) >= stability_bound:
                # 1. Confidence-Weighted Exact String Voting
                # Instead of naive char padding, find the string variant within the cluster
                # that has the highest accumulated confidence.
                score_map = defaultdict(float)
                for t, c in cluster:
                    score_map[t] += c
                
                # Pick the text with the highest score
                best_text = max(score_map.items(), key=lambda x: x[1])[0]
                
                # 2. Positional Heuristics
                import re
                clean = re.sub(r'[^A-Z0-9]', '', best_text)
                chars_list = list(clean)
                n = len(chars_list)
                
                dict_char_to_int = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'Q': '0', 'D': '0', 'A': '4'}
                dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A'}
                
                for i in range(min(n, 2)):
                    if chars_list[i].isdigit() and chars_list[i] in dict_int_to_char:
                        chars_list[i] = dict_int_to_char[chars_list[i]]
                
                for i in range(2, min(n, 4)):
                    if chars_list[i].isalpha() and chars_list[i] in dict_char_to_int:
                        chars_list[i] = dict_char_to_int[chars_list[i]]
                        
                if n >= 8:
                    for i in range(max(4, n-4), n):
                        if chars_list[i].isalpha() and chars_list[i] in dict_char_to_int:
                            chars_list[i] = dict_char_to_int[chars_list[i]]
                            
                self.final_text = "".join(chars_list)
                self.final_conf = sum(c for t, c in cluster) / len(cluster)
                self.status = "locked"
                
                log.info(f"STABILITY REACHED (Weighted): {self.final_text} from {best_text} (Cluster size: {len(cluster)})")
                return True
                
        return False

class ANPRController:
    def __init__(self, config):
        self.config = config
        
        # Init Sub-systems
        self.detector = PlateDetector(
            model_path=config['detection']['model_path'],
            tracker_config=config['tracking']['tracker_config'],
            conf_thresh=config['detection']['conf_threshold'],
            iou_thresh=config['detection']['iou_threshold'],
            img_size=config['detection']['img_size']
        )
        
        self.ocr = OCRProcessor(
            lang=config['ocr']['lang'],
            conf_thresh=config['ocr']['conf_threshold'],
            regex_pattern=config['ocr']['regex_pattern']
        )
        
        # Tracking State
        self.active_tracks = {} # track_id -> VehicleTrack
        self.plate_tracks = defaultdict(list)
        self.finalized_tracks = {}
        self.latest_plate = {}
        self.final_plates = []
        self.last_ocr_frame = {}
        self.output_root = config.get("output", {}).get("root_dir", "output")
        self.dataset_enabled = config.get("dataset", {}).get("enabled", True)
        self.detected_track_ids = set()
        self.read_track_ids = set()
        
        # Global Event Registry (for deduplication)
        # {plate_text: last_event_timestamp}
        self.unique_plate_registry = {} 
        self.dedup_timeout = config['database']['deduplication_timeout_s']
        
        self.stability_bound = config['ocr']['stability_frames']
        self.buffer_size = 5 # Strict cap

        # Perf stats
        self.frame_count = 0
        self.ocr_calls = 0
        self.last_metric_time = time.time()
        self.interval_ocr_calls = 0

    def add_plate(self, new_plate, conf):
        for i, (existing_plate, existing_conf) in enumerate(self.final_plates):
            if is_similar_plate(existing_plate, new_plate):
                if conf > existing_conf:
                    self.final_plates[i] = (new_plate, conf)
                    return new_plate, conf
                return existing_plate, existing_conf

        self.final_plates.append((new_plate, conf))
        return new_plate, conf

    def get_final_plates(self):
        return sorted(self.final_plates, key=lambda item: item[1], reverse=True)

    def get_stats(self):
        return {
            "plates_detected": len(self.detected_track_ids),
            "plates_read": len(self.read_track_ids),
        }

    def run_ocr(self, crop, track_id, vehicle):
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        raw_text, ocr_conf = self.ocr.recognize(crop)
        if not raw_text:
            return

        cleaned = clean_plate(raw_text)
        if len(cleaned) < 8 or len(cleaned) > 12:
            return
        final_text = correct_plate(cleaned)
        if final_text is None:
            return

        self.ocr_calls += 1
        self.interval_ocr_calls += 1
        self.latest_plate[track_id] = (final_text, ocr_conf)

        if ocr_conf > 0.15:
            self.plate_tracks[track_id].append((final_text, ocr_conf, self.frame_count))
            vehicle.add_ocr(final_text, ocr_conf, crop)
            self.read_track_ids.add(track_id)
            print(f"[TRACK {track_id}] OCR: {final_text} | CONF: {ocr_conf:.2f}")
            log.info(f"[TRACK {track_id}] OCR: {final_text} | CONF: {ocr_conf:.2f}")

    def process_frame(self, frame):
        """
        Main pipeline step.
        """
        frame = cv2.resize(frame, (960, 540))
        self.frame_count += 1
        current_time = time.time()
        new_events = []
        
        if not hasattr(self, 'prev_time'): self.prev_time = current_time - 0.03
        fps = 1 / max(1e-5, current_time - self.prev_time)
        self.prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Periodic Metrics Print (Every 2 seconds)
        if current_time - self.last_metric_time >= 2.0:
            cps = self.interval_ocr_calls / (current_time - self.last_metric_time)
            log.info(f"VALUE TRACE | OCR_CALL_COUNT (per second): {cps:.2f}")
            self.interval_ocr_calls = 0
            self.last_metric_time = current_time
        
        tracks = self.detector.detect_and_track(frame)
        
        # 2. Update Active Tracks
        seen_ids = set()
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls_id = track
            seen_ids.add(track_id)
            self.detected_track_ids.add(track_id)
            
            # Create new track if needed
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = VehicleTrack(track_id, current_time)
                # log.debug(f"New Track: {track_id}")
            
            vehicle = self.active_tracks[track_id]
            vehicle.last_seen = current_time
            vehicle.total_frames_alive += 1
            
            # Visualization logic per requirements
            raw_reads = list(self.plate_tracks.get(track_id, []))
            final_plate, confidence = final_voting(raw_reads) if raw_reads else (None, 0.0)
            
            if final_plate:
                display_text = f"{final_plate} ({confidence:.2f})"
            else:
                display_text = ""

            track_color = (0, 255, 0)
            
            if final_plate and confidence >= 0.7:
                vehicle.final_text = final_plate
                vehicle.final_conf = confidence
                vehicle.final_text, vehicle.final_conf = self.add_plate(vehicle.final_text, vehicle.final_conf)
                
                # Check Global Registry FIRST
                if self._should_save_event(vehicle.final_text, current_time, vehicle):
                    self._save_event(vehicle, current_time)
                    new_events.append({
                        'plate_number': vehicle.final_text,
                        'confidence': vehicle.final_conf,
                        'track_id': track_id,
                        'timestamp': current_time,
                        'crop': vehicle.best_crop
                    })
                    
                # Save to Dataset (Once per Locked Track)
                if self.dataset_enabled and not getattr(vehicle, 'dataset_saved', False):
                    self._save_to_dataset(vehicle)
                    vehicle.dataset_saved = True
                    print(f"[TRACK {track_id}] FINAL: {vehicle.final_text} | CONF: {vehicle.final_conf:.2f}")

            # --- INTELLIGENT OCR GATING ---
            run_ocr_flag = self.frame_count % 5 == 0
            if run_ocr_flag:
                current_frame = self.frame_count
                last_frame = self.last_ocr_frame.get(track_id)
                if last_frame is not None and current_frame - last_frame < 10:
                    run_ocr_flag = False

            if run_ocr_flag:
                bw, bh = x2 - x1, y2 - y1
                
                pad_w = int(bw * 0.1)
                pad_h = int(bh * 0.1)
                cx1, cy1 = max(0, int(x1) - pad_w), max(0, int(y1) - pad_h)
                cx2, cy2 = min(frame.shape[1], int(x2) + pad_w), min(frame.shape[0], int(y2) + pad_h)
                
                plate_crop = frame[cy1:cy2, cx1:cx2]
                
                if plate_crop.size > 0:
                    crop_h, crop_w = plate_crop.shape[:2]
                    if crop_w >= 80 and crop_h >= 25:
                        self.run_ocr(plate_crop, track_id, vehicle)
                        self.last_ocr_frame[track_id] = current_frame

            # --- DRAWING ---
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_color, 2)
            
            latest = self.latest_plate.get(track_id)
            if latest:
                plate, conf = latest
                if conf > 0.85:
                    display_text = f"{plate} ({conf:.2f})"
                    cv2.putText(frame, display_text, (int(x1), int(max(0, y1-10))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 3. Cleanup Lost Tracks
        self._cleanup_tracks(current_time, seen_ids)
        
        return frame, new_events

    def _should_save_event(self, plate_text, current_time, vehicle=None):
        """
        Enforce One Event Per Plate Rule and Strict Regex.
        """
        if not self.ocr.strict_check(plate_text):
            log.warning(f"VALUE TRACE | {plate_text} rejected by Strict Regex.")
            return False
            
        for saved_plate, last_saved in self.unique_plate_registry.items():
            if is_similar_plate(saved_plate, plate_text) and (current_time - last_saved) <= self.dedup_timeout:
                return False

        last_saved = self.unique_plate_registry.get(plate_text, 0)
        
        # If never seen OR timeout exceeded
        if (current_time - last_saved) > self.dedup_timeout:
            return True
            
        return False

    def _save_event(self, vehicle, current_time):
        """
        Register the event to global registry.
        """
        self.unique_plate_registry[vehicle.final_text] = current_time
        log.info(f"EVENT SAVED: {vehicle.final_text} (Track {vehicle.track_id})")

    def _save_to_dataset(self, vehicle):
        """
        Save rectified plate crop and label for future training.
        """
        try:
            if vehicle.best_crop is None:
                return

            import csv
            
            # Dirs
            img_dir = os.path.join(self.output_root, "dataset", "ocr_training", "images")
            os.makedirs(img_dir, exist_ok=True)
            csv_path = os.path.join(self.output_root, "dataset", "ocr_training", "labels.csv")
            
            # Rectify before saving (using the new OCR helper)
            rectified = self.ocr.rectify_plate(vehicle.best_crop)
            
            # Filename
            timestamp = int(time.time() * 1000)
            filename = f"plate_{vehicle.track_id}_{timestamp}.jpg"
            save_path = os.path.join(img_dir, filename)
            
            # Save Image
            cv2.imwrite(save_path, rectified)
            
            # Append to Dataset CSV
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['image_name', 'text', 'confidence'])
                writer.writerow([filename, vehicle.final_text, f"{vehicle.final_conf:.4f}"])
            
            # Additional User requirement: events.csv
            events_csv = os.path.join(self.output_root, "dataset", "events.csv")
            events_exists = os.path.isfile(events_csv)
            with open(events_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not events_exists:
                    writer.writerow(['plate_number', 'confidence', 'timestamp', 'camera_id', 'image_path'])
                # camera_id simplified as 'cam_0' for local testing
                writer.writerow([vehicle.final_text, f"{vehicle.final_conf:.4f}", timestamp, 'cam_0', save_path])

            log.info(f"Saved to DB. (events.csv)")
            
        except Exception as e:
            log.error(f"Failed to save dataset entry: {e}")

    def finalize_plate(self, track_id):
        if track_id in self.finalized_tracks:
            return None

        raw_reads = list(self.plate_tracks.get(track_id, []))
        if not raw_reads: return None
        plate, conf = final_voting(raw_reads)
        
        if plate is None:
            return None
            
        self.finalized_tracks[track_id] = plate
        return plate

    def _cleanup_tracks(self, current_time, seen_ids):
        """
        Remove tracks not seen in this frame (with short grace period?).
        For simplicity, strict removal if not seen for X seconds.
        """
        timeout = 2.0 # Keep short
        to_remove = []
        
        for tid, vehicle in self.active_tracks.items():
            if tid not in seen_ids:
                if (current_time - vehicle.last_seen) > timeout:
                    to_remove.append(tid)
        
        for tid in to_remove:
            self.active_tracks.pop(tid, None)
            self.plate_tracks.pop(tid, None)
            self.latest_plate.pop(tid, None)
            self.finalized_tracks.pop(tid, None)
            self.last_ocr_frame.pop(tid, None)

        for track_id in list(self.plate_tracks.keys()):
            if track_id not in self.active_tracks:
                self.plate_tracks.pop(track_id, None)
                self.latest_plate.pop(track_id, None)
                self.finalized_tracks.pop(track_id, None)
                self.last_ocr_frame.pop(track_id, None)

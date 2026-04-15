
import cv2
import re
import numpy as np
import os
import sys


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ.setdefault("FLAGS_cpu_math_library_num_threads", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR
from src.utils.logger import log

class OCRProcessor:
    def __init__(self, lang='en', conf_thresh=0.4, regex_pattern=None):
        self.conf_thresh = conf_thresh
        # Indian Plate Pattern: 2 Letters, 1-2 Digits, 1-2 Letters, 4 Digits
        # e.g., MH 12 AB 1234
        self.regex_pattern = regex_pattern or r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$"
        
        log.info("Initializing PaddleOCR...")
        # gpu=True to use GPU
        import logging
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        # Disable MKL-DNN to fix the C++ NotImplementedError in onednn_instruction.cc
        custom_model_dir = "output/custom_anpr_v4_mobile/infer"
        custom_dict_path = "dataset/ocr_dataset/indian_plate_dict.txt"
        
        if os.path.exists(custom_model_dir):
            log.info(f"Using fine-tuned model from {custom_model_dir}")
            reader_kwargs = dict(
                lang='en',
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                rec_model_dir=custom_model_dir
            )
            if os.path.exists(custom_dict_path):
                reader_kwargs["rec_char_dict_path"] = custom_dict_path
            self.reader = PaddleOCR(**reader_kwargs)
        else:
            log.info("Fine-tuned model not found. Using default pre-trained model.")
            self.reader = PaddleOCR(
                lang='en',
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
        log.info("PaddleOCR Initialized.")

    def preprocess_variants(self, crop):
        variants = []

        if crop is None or crop.size == 0:
            return variants

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        variants.append(gray)

        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        variants.append(thresh)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        variants.append(blur)

        return variants

    def preprocess_image(self, crop):
        h, w = crop.shape[:2]

        if w < 150:
            crop = cv2.resize(crop, (w*2, h*2))

        # Convert to Grayscale + Enhance
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        gray = cv2.equalizeHist(gray)
        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Apply Sharpening
        kernel = [[0, -1, 0],
                  [-1, 5,-1],
                  [0, -1, 0]]
        kernel = np.array(kernel)
        crop = cv2.filter2D(crop, -1, kernel)

        return crop

    def rect_to_box(self, rect):
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    def rectify_plate(self, image):
        return image

    def upscale_plate(self, img):
        """
        Resize to target 48px height for optimal PaddleOCR feature extraction.
        """
        h, w = img.shape[:2]
        if h != 48 and h > 0:
            scale = 48 / h
            new_w = max(1, int(w * scale))
            new_h = 48
            upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return upscaled
        return img

    def _extract_text_from_result(self, result):
        try:
            res = result[0]
            if isinstance(res, dict) and 'rec_texts' in res:
                text = res['rec_texts'][0]
                conf = res['rec_scores'][0]
            elif hasattr(res, 'keys') and 'rec_texts' in res:
                text = res['rec_texts'][0]
                conf = res.get('rec_scores', [0.0])[0]
            elif hasattr(res, 'rec_texts'):
                text = res.rec_texts[0]
                conf = getattr(res, 'rec_scores', [0.0])[0]
            else:
                text = result[0][0][1][0]
                conf = result[0][0][1][1]
            return text, conf
        except Exception:
            return None, 0.0

    def _run_ocr_once(self, image):
        if image is None or image.size == 0:
            return None, 0.0

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        result = self.reader.ocr(image)
        return self._extract_text_from_result(result)

    def robust_ocr(self, plate_crop):
        best_text = None
        best_conf = 0.0

        base_variant = self.preprocess_image(plate_crop)
        candidates = [base_variant]
        for variant in self.preprocess_variants(plate_crop):
            if len(variant.shape) == 2:
                variant = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
            candidates.append(variant)

        for candidate in candidates:
            text, conf = self._run_ocr_once(candidate)
            if not text:
                continue

            cleaned_text = re.sub(r'[^A-Z0-9]', '', str(text).upper())
            validity_bonus = 0.1 if 8 <= len(cleaned_text) <= 12 else 0.0
            score = conf + validity_bonus
            best_score = best_conf + (0.1 if best_text and 8 <= len(re.sub(r'[^A-Z0-9]', '', str(best_text).upper())) <= 12 else 0.0)

            if score > best_score:
                best_text = text
                best_conf = conf

        return best_text, best_conf

    def recognize(self, plate_crop, debug_id=None):
        """
        Run OCR on text image.
        Returns: (text, confidence)
        """
        try:
            if plate_crop is None or plate_crop.size == 0:
                return None, 0.0

            processed = self.preprocess_image(plate_crop)

            if debug_id:
                debug_dir = "data/debug_crops"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/{debug_id}_proc.jpg", processed)
                cv2.imwrite(f"{debug_dir}/{debug_id}_raw.jpg", plate_crop)

            text, conf = self._run_ocr_once(processed)
            cleaned = re.sub(r'[^A-Z0-9]', '', str(text).upper()) if text else ""
            if not text or conf < self.conf_thresh or len(cleaned) < 8:
                text, conf = self.robust_ocr(plate_crop)

            return text, conf

        except Exception as e:
            import traceback
            log.error(f"OCR Failed: {e}\n{traceback.format_exc()}")
            return None, 0.0


    def smart_correct(self, text):
        """
        Apply positional corrections based on Indian License Plate format.
        General Pattern: LL DD LL DDDD (e.g., MH 12 DE 1433)
        Heuristic:
        - First 2 chars -> Letters
        - Next 2 chars -> Digits
        - Last 4 chars -> Digits
        """
        if not text: 
            return ""
            
        text = text.upper().replace(' ', '')
        chars = list(text)
        n = len(chars)
        
        # Mappings
        dict_char_to_int = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'Q': '0', 'D': '0', 'A': '4'}
        dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A'}
        
        # Helper to swap
        def to_int(idx):
            if idx < n and chars[idx] in dict_char_to_int:
                chars[idx] = dict_char_to_int[chars[idx]]
                
        def to_char(idx):
            if idx < n and chars[idx] in dict_int_to_char:
                chars[idx] = dict_int_to_char[chars[idx]]

        # 1. First 2 chars should be Letters (State Code e.g. MH, DL)
        for i in range(min(n, 2)):
            if chars[i].isdigit():
                to_char(i)
                
        # 2. Next 2 chars should be Digits (District Code e.g. 12, 01)
        for i in range(2, min(n, 4)):
            if chars[i].isalpha():
                to_int(i)
                
        # 3. Last 4 chars should be Digits (Number e.g. 1433)
        # Only if length suggests full plate (>= 8 chars)
        if n >= 8:
            for i in range(max(4, n-4), n):
                if chars[i].isalpha():
                    to_int(i)
                    
        return "".join(chars)

    def validate(self, text):
        """
        Soft validation and correction. 
        Returns (is_valid_candidate, corrected_text)
        """
        if not text:
            return False, ""
            
        # 1. Basic cleaning
        import re
        # Allow A-Z, 0-9
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # 2. Smart correction based on Indian Plate pos
        corrected = self.smart_correct(clean)
        
        # 3. Validation Logic (Length Check)
        if len(corrected) < 4:
            return False, corrected
            
        return True, corrected

    def strict_check(self, text):
        """
        Strict regex validation for final decision.
        """
        if self.regex_pattern:
            import re
            return bool(re.match(self.regex_pattern, text))
        return True

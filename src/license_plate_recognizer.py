import cv2
import numpy as np
import easyocr
import torch
import re
import pytesseract
from collections import deque
from datetime import datetime
from typing import Optional, Tuple
from ultralytics import YOLO


class LicensePlateRecognizer:
    """
    License plate detection and recognition system.
    Uses YOLO for plate detection and EasyOCR for reading (with enhanced preprocessing).
    Designed specifically for Camera 1 (entrance camera) only.
    """

    def __init__(self, plate_model_path='src/models/plate_detector/best.pt',
                 plate_detection_confidence=0.5,
                 gpu_enabled=None,
                 strict_validation=False,
                 aggregation_window=25,
                 **kwargs):
        """
        Initialize the license plate recognizer.

        Args:
            plate_model_path: Path to YOLO plate detection model
            plate_detection_confidence: Confidence threshold for plate detection
            gpu_enabled: If None, auto-detect GPU availability
            strict_validation: If True, only accept plates matching Indian format exactly
        """
        # YOLO plate detector
        self.plate_detector = YOLO(plate_model_path)
        self.plate_detection_confidence = plate_detection_confidence
        print(f"[LPR] YOLO plate detector loaded: {plate_model_path}")

        # EasyOCR reader
        if gpu_enabled is None:
            gpu_enabled = torch.cuda.is_available()
        self.reader = easyocr.Reader(['en'], gpu=gpu_enabled)
        print(f"[LPR] EasyOCR initialized (GPU: {gpu_enabled})")

        # Multi-frame aggregation (number of frames held for majority voting)
        self.plate_history = {}
        self.history_window = aggregation_window

        # Validation mode
        self.strict_validation = strict_validation

        # Reusable CLAHE instance (avoids creating one per frame)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Probe whether the Tesseract binary is installed
        self._tesseract_available = False
        try:
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            print("[LPR] Tesseract OCR available")
        except Exception:
            print("[LPR] Tesseract OCR not found — skipping Tesseract reads")

        print(f"[LPR] Initialized (Strict validation: {strict_validation})")

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return self._clahe.apply(img)

    def _unsharp_mask(self, img: np.ndarray, strength: float = 1.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    def detect_plate_region(self, vehicle_crop):
        """
        Detect license plate regions within a vehicle crop using YOLO.

        Args:
            vehicle_crop: Cropped image of detected vehicle (BGR)

        Returns:
            List of plate regions as (x, y, w, h) tuples
        """
        results = self.plate_detector(
            vehicle_crop,
            conf=self.plate_detection_confidence,
            verbose=False
        )

        plate_regions = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                if w >= 20 and h >= 8:
                    plate_regions.append((x1, y1, w, h))

        return plate_regions

    def _enhance_plate(self, plate_img: np.ndarray) -> dict:
        """
        Returns three preprocessed variants targeting different RTSP failure modes:
          - 'colour'     : bilateral denoise → CLAHE (LAB) → 4x upscale → unsharp mask
          - 'gray_clahe' : grayscale → bilateral denoise → CLAHE → 4x upscale → unsharp mask
          - 'binary'     : grayscale → bilateral denoise → 4x upscale → adaptive threshold
        """
        h, w = plate_img.shape[:2]
        target = (w * 4, h * 4)

        # --- Variant 1: colour + CLAHE ---
        denoised = cv2.bilateralFilter(plate_img, 9, 75, 75)
        clahe_colour = self._apply_clahe(denoised)
        big_colour = cv2.resize(clahe_colour, target, interpolation=cv2.INTER_CUBIC)
        colour_var = self._unsharp_mask(big_colour)

        # --- Variant 2: grayscale + CLAHE ---
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray_denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        gray_clahe = self._clahe.apply(gray_denoised)
        big_gray = cv2.resize(gray_clahe, target, interpolation=cv2.INTER_CUBIC)
        gray_var = self._unsharp_mask(big_gray)

        # --- Variant 3: adaptive binary threshold ---
        big_gray_raw = cv2.resize(gray_denoised, target, interpolation=cv2.INTER_CUBIC)
        binary_var = cv2.adaptiveThreshold(
            big_gray_raw, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            11, 2
        )

        return {'colour': colour_var, 'gray_clahe': gray_var, 'binary': binary_var}

    # Restrict EasyOCR output to plate-valid characters — eliminates lowercase /
    # punctuation / non-Latin confusions that pollute the voting pool.
    _OCR_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    def _tesseract_ocr(self, img: np.ndarray, psm: int = 7) -> Tuple[Optional[str], float]:
        """
        Run Tesseract on a plate image. Returns (cleaned_text, confidence 0–1).
        Silently returns (None, 0.0) if Tesseract binary is not installed.
        """
        if not self._tesseract_available:
            return None, 0.0
        try:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            texts, confs = [], []
            for text, conf in zip(data['text'], data['conf']):
                text = text.strip()
                if text and int(conf) > 0:
                    texts.append(text)
                    confs.append(int(conf))
            if texts:
                cleaned = self.clean_plate_text(''.join(texts))
                if cleaned:
                    return cleaned, sum(confs) / len(confs) / 100.0
        except Exception:
            pass
        return None, 0.0

    def _ocr_plate(self, plate_img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Multi-variant EasyOCR + Tesseract ensemble.
        Runs up to 5 reads (3 EasyOCR variants + 2 Tesseract configs) and returns
        the highest-confidence result. Falls back to original image if all fail.
        """
        best_text: Optional[str] = None
        best_conf: float = 0.0

        def _update(text, conf):
            nonlocal best_text, best_conf
            if text and conf > best_conf:
                best_text, best_conf = text, conf

        variants = self._enhance_plate(plate_img)

        # --- EasyOCR on all 3 variants ---
        for variant_img in variants.values():
            try:
                results = self.reader.readtext(variant_img, detail=1, allowlist=self._OCR_ALLOWLIST)
                for (_, text, conf) in results:
                    _update(self.clean_plate_text(text), conf)
            except Exception:
                pass

        # --- Tesseract: psm 7 (single line) on gray_clahe ---
        _update(*self._tesseract_ocr(variants['gray_clahe'], psm=7))

        # --- Tesseract: psm 8 (single word) on binary ---
        _update(*self._tesseract_ocr(variants['binary'], psm=8))

        # --- Fallback: EasyOCR on original image ---
        if best_text is None or best_conf < 0.3:
            try:
                results = self.reader.readtext(plate_img, detail=1, allowlist=self._OCR_ALLOWLIST)
                for (_, text, conf) in results:
                    _update(self.clean_plate_text(text), conf)
            except Exception:
                pass

        return best_text, best_conf

    def clean_plate_text(self, text):
        """
        Clean and normalize the detected license plate text.
        """
        if not text:
            return None

        # Remove all non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Filter out very short results (likely noise)
        if len(cleaned) < 3:
            return None

        # Filter out very long results (likely false detection)
        if len(cleaned) > 15:
            return None

        return cleaned

    def validate_indian_plate_format(self, text: str) -> Tuple[str, bool]:
        """
        Validate and correct Indian plate format: KL01AA3456
        Pattern: 2 letters + 2 digits + 1-2 letters + 4 digits

        Returns: (corrected_text, is_valid)
        """
        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        indian_patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # 10 chars: KL01AA3456
            r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$',   # 9 chars:  KL01A3456
        ]

        for pattern in indian_patterns:
            if re.match(pattern, text):
                return text, True

        corrected = self.attempt_format_correction(text)
        if corrected:
            return corrected, True

        return text, False

    def attempt_format_correction(self, text: str) -> Optional[str]:
        """
        Correct common OCR errors based on Indian plate format.
        """
        if len(text) < 9 or len(text) > 10:
            return None

        corrected = list(text)

        # Common EasyOCR confusions on Indian plates. Expanded beyond the original
        # five pairs to cover G/6, A/4, T/7, D/0, Q/0, L/1 — all frequent misreads.
        digit_to_letter = {
            '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S',
            '6': 'G', '7': 'T', '8': 'B',
        }
        letter_to_digit = {
            'O': '0', 'D': '0', 'Q': '0',
            'I': '1', 'L': '1',
            'Z': '2',
            'A': '4',
            'S': '5',
            'G': '6',
            'T': '7',
            'B': '8',
        }

        # Positions 0-1: Force letters
        for i in [0, 1]:
            if corrected[i].isdigit():
                corrected[i] = digit_to_letter.get(corrected[i], corrected[i])

        # Positions 2-3: Force digits
        for i in [2, 3]:
            if corrected[i].isalpha():
                corrected[i] = letter_to_digit.get(corrected[i], corrected[i])

        # Position 4 (and 5 if exists): Force letters
        for i in [4, 5]:
            if i < len(corrected) and corrected[i].isdigit():
                corrected[i] = digit_to_letter.get(corrected[i], corrected[i])

        # Last 4 positions: Force digits
        start_idx = len(corrected) - 4
        for i in range(start_idx, len(corrected)):
            if corrected[i].isalpha():
                corrected[i] = letter_to_digit.get(corrected[i], corrected[i])

        result = ''.join(corrected)

        for pattern in [r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$']:
            if re.match(pattern, result):
                return result

        return None

    def aggregate_plate_readings(self, vehicle_id: str, plate: str, confidence: float) -> Tuple[str, float]:
        """
        Aggregate plate readings across multiple frames using voting.

        Only plates that pass Indian-format validation (after character-confusion
        correction) are added to the voting pool. Unvalidated reads are returned
        as-is but do NOT pollute the history, so voting converges on the true plate.

        Returns: (best_plate, aggregated_confidence)
        """
        # Normalize through format validation / correction before voting.
        validated, is_valid = self.validate_indian_plate_format(plate)

        if is_valid:
            plate = validated
            if vehicle_id not in self.plate_history:
                self.plate_history[vehicle_id] = deque(maxlen=self.history_window)
            self.plate_history[vehicle_id].append({
                'plate': plate,
                'confidence': confidence,
                'timestamp': datetime.now()
            })

        history = self.plate_history.get(vehicle_id)
        if not history or len(history) < 3:
            return plate, confidence

        readings = list(history)
        plates = [r['plate'] for r in readings]
        confidences = [r['confidence'] for r in readings]

        plate_scores = {}
        for p, conf in zip(plates, confidences):
            plate_scores[p] = plate_scores.get(p, 0) + conf

        best_plate = max(plate_scores, key=plate_scores.get)

        best_confidences = [conf for p, conf in zip(plates, confidences) if p == best_plate]
        aggregated_conf = sum(best_confidences) / len(best_confidences)

        agreement_ratio = plates.count(best_plate) / len(plates)
        if agreement_ratio >= 0.6:
            aggregated_conf = min(1.0, aggregated_conf * 1.2)

        return best_plate, aggregated_conf

    def recognize_plate(self, vehicle_crop, min_confidence=0.5):
        """
        Recognize license plate number from vehicle crop.

        Pipeline: YOLO detect plate region -> crop -> enhance -> EasyOCR -> validate

        Args:
            vehicle_crop: Cropped image of detected vehicle (BGR)
            min_confidence: Minimum confidence threshold for results

        Returns:
            Tuple (plate_number, confidence, plate_bbox)
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, 0.0, None

        if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
            return None, 0.0, None

        best_plate = None
        best_confidence = 0.0
        best_bbox = None

        # Phase 1: YOLO plate detection -> crop -> enhanced OCR
        plate_regions = self.detect_plate_region(vehicle_crop)

        for (x, y, w, h) in plate_regions:
            # Extract plate region with minimal padding
            pad_x = max(2, int(w * 0.02))
            pad_y = max(2, int(h * 0.02))
            y1 = max(0, y - pad_y)
            y2 = min(vehicle_crop.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(vehicle_crop.shape[1], x + w + pad_x)

            plate_roi = vehicle_crop[y1:y2, x1:x2]

            if plate_roi.size == 0:
                continue

            text, conf = self._ocr_plate(plate_roi)

            if text and conf > best_confidence:
                best_plate = text
                best_confidence = conf
                best_bbox = (x, y, w, h)

            if best_confidence >= 0.7:
                break

        # Phase 2: Fallback - send lower half of vehicle crop
        if best_plate is None or best_confidence < min_confidence:
            height = vehicle_crop.shape[0]
            lower_half = vehicle_crop[int(height * 0.4):, :]

            if lower_half.size > 0:
                text, conf = self._ocr_plate(lower_half)
                if text and conf > best_confidence:
                    best_plate = text
                    best_confidence = conf
                    best_bbox = None

        # Phase 3: Validate with Indian format
        if best_plate and best_confidence >= min_confidence:
            validated_plate, is_valid = self.validate_indian_plate_format(best_plate)

            if is_valid:
                if validated_plate != best_plate:
                    print(f"[LPR] Validated Indian plate: {validated_plate} (original: {best_plate})")
                return validated_plate, best_confidence, best_bbox
            else:
                if not self.strict_validation:
                    return best_plate, best_confidence, best_bbox
                else:
                    print(f"[LPR] Invalid format rejected: {best_plate}")
                    return None, 0.0, None

        return None, 0.0, None

    def batch_recognize(self, vehicle_crops, min_confidence=0.5):
        """
        Recognize license plates from multiple vehicle crops (batch processing).
        """
        results = []
        for crop in vehicle_crops:
            plate_info = self.recognize_plate(crop, min_confidence)
            results.append(plate_info)
        return results

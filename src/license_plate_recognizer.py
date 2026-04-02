import cv2
import numpy as np
import easyocr
import torch
import re
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

        # Multi-frame aggregation
        self.plate_history = {}
        self.history_window = 15  # frames (~0.5 sec at 30fps)

        # Validation mode
        self.strict_validation = strict_validation

        print(f"[LPR] Initialized (Strict validation: {strict_validation})")

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

    def _enhance_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Enhance plate crop for OCR: 4x upscale + sharpen.
        Dramatically improves EasyOCR accuracy on small plate crops.
        """
        h, w = plate_img.shape[:2]
        # 4x upscale with cubic interpolation
        big = cv2.resize(plate_img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp = cv2.filter2D(big, -1, kernel)
        return sharp

    def _ocr_plate(self, plate_img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Run EasyOCR on a plate image with enhanced preprocessing.
        Tries enhanced image first, then original as fallback.

        Returns:
            Tuple (best_text, best_confidence)
        """
        best_text = None
        best_conf = 0.0

        # Strategy 1: 4x upscaled + sharpened (best for small plates)
        enhanced = self._enhance_plate(plate_img)
        try:
            results = self.reader.readtext(enhanced, detail=1)
            for (bbox, text, conf) in results:
                cleaned = self.clean_plate_text(text)
                if cleaned and conf > best_conf:
                    best_text = cleaned
                    best_conf = conf
        except Exception:
            pass

        # Strategy 2: Original image (fallback for already-large plates)
        if best_text is None or best_conf < 0.3:
            try:
                results = self.reader.readtext(plate_img, detail=1)
                for (bbox, text, conf) in results:
                    cleaned = self.clean_plate_text(text)
                    if cleaned and conf > best_conf:
                        best_text = cleaned
                        best_conf = conf
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

        digit_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z'}
        letter_to_digit = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2'}

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

        Returns: (best_plate, aggregated_confidence)
        """
        if vehicle_id not in self.plate_history:
            self.plate_history[vehicle_id] = deque(maxlen=self.history_window)

        self.plate_history[vehicle_id].append({
            'plate': plate,
            'confidence': confidence,
            'timestamp': datetime.now()
        })

        if len(self.plate_history[vehicle_id]) < 3:
            return plate, confidence

        readings = list(self.plate_history[vehicle_id])
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

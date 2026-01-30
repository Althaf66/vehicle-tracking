import cv2
import numpy as np
import easyocr
import torch
import re
from collections import deque
from datetime import datetime
from typing import Optional, Tuple


class LicensePlateRecognizer:
    """
    License plate detection and recognition system using EasyOCR.
    Designed specifically for Camera 1 (entrance camera) only.
    """

    def __init__(self, gpu_enabled=None, strict_validation=False):
        """
        Initialize the license plate recognizer.

        Args:
            gpu_enabled: If None, auto-detect GPU availability. Otherwise, use the provided boolean.
            strict_validation: If True, only accept plates matching Indian format exactly
        """
        if gpu_enabled is None:
            gpu_enabled = torch.cuda.is_available()

        # Initialize EasyOCR reader for license plate recognition
        # Using English language model for alphanumeric plates
        self.reader = easyocr.Reader(['en'], gpu=gpu_enabled)

        # Multi-frame aggregation
        self.plate_history = {}  # vehicle_id -> deque of (plate, confidence, timestamp)
        self.history_window = 15  # frames (~0.5 sec at 30fps)

        # Validation mode
        self.strict_validation = strict_validation

        print(f"License Plate Recognizer initialized (GPU: {gpu_enabled}, Strict validation: {strict_validation})")

    def preprocess_plate_region(self, plate_img):
        """
        Preprocess the license plate region to improve OCR accuracy.

        Args:
            plate_img: BGR image of the license plate region

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)

        # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Apply adaptive thresholding to get binary image
        # This helps with varying lighting conditions
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def preprocess_indian_plate(self, plate_img):
        """
        Specialized preprocessing for Indian license plates
        Handles yellow/white backgrounds with black text

        Returns: (binary_otsu, binary_adaptive)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 1. Morphological operations to enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # 2. Bilateral filtering
        denoised = cv2.bilateralFilter(morph, 11, 17, 17)

        # 3. Enhanced CLAHE for Indian plates (higher clip limit)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 4. Try multiple binarization methods
        # Method 1: Otsu's thresholding (good for yellow plates)
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Method 2: Adaptive threshold (fallback)
        binary_adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Return both - try OCR on both and pick best result
        return binary_otsu, binary_adaptive

    def clean_plate_text(self, text):
        """
        Clean and normalize the detected license plate text.

        Args:
            text: Raw text from OCR

        Returns:
            Cleaned plate number (alphanumeric only, uppercase)
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
        # Remove spaces and special characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Standard Indian format: XX00XX0000 or XX00X0000
        indian_patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # 10 chars: KL01AA3456
            r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$',  # 9 chars:  KL01A3456
        ]

        # Check if matches any pattern
        for pattern in indian_patterns:
            if re.match(pattern, text):
                return text, True

        # Attempt correction if close to valid format
        corrected = self.attempt_format_correction(text)
        if corrected:
            return corrected, True

        return text, False

    def attempt_format_correction(self, text: str) -> Optional[str]:
        """
        Correct common OCR errors based on Indian plate format

        Rules:
        - Positions 0-1: Must be letters (A-Z)
        - Positions 2-3: Must be digits (0-9)
        - Positions 4-5: Must be letters (A-Z)
        - Positions 6-9: Must be digits (0-9)
        """
        if len(text) < 9 or len(text) > 10:
            return None

        corrected = list(text)

        # Define confusion mappings
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

        # Validate corrected format
        for pattern in [r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$']:
            if re.match(pattern, result):
                return result

        return None

    def detect_plate_region(self, vehicle_crop):
        """
        Detect potential license plate regions within a vehicle crop.
        Uses aspect ratio and position heuristics.

        Args:
            vehicle_crop: Cropped image of detected vehicle

        Returns:
            List of potential plate regions (x, y, w, h) or empty list
        """
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter
        blur = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edges = cv2.Canny(blur, 30, 200)

        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plate_candidates = []

        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # License plate aspect ratio is typically between 2:1 and 5:1
            aspect_ratio = w / float(h) if h > 0 else 0

            # Filter based on aspect ratio and size
            if 2.0 <= aspect_ratio <= 5.5:
                # Area should be reasonable (not too small, not too large)
                area = w * h
                vehicle_area = vehicle_crop.shape[0] * vehicle_crop.shape[1]
                area_ratio = area / vehicle_area

                # Plate should be between 1% and 20% of vehicle area
                if 0.01 <= area_ratio <= 0.20:
                    # Plate is usually in the lower half of the vehicle
                    vehicle_height = vehicle_crop.shape[0]
                    if y > vehicle_height * 0.3:  # Lower 70% of vehicle
                        plate_candidates.append((x, y, w, h))

        return plate_candidates

    def aggregate_plate_readings(self, vehicle_id: str, plate: str, confidence: float) -> Tuple[str, float]:
        """
        Aggregate plate readings across multiple frames using voting

        Returns: (best_plate, aggregated_confidence)
        """
        from collections import Counter

        # Initialize history for new vehicle
        if vehicle_id not in self.plate_history:
            self.plate_history[vehicle_id] = deque(maxlen=self.history_window)

        # Add current reading
        self.plate_history[vehicle_id].append({
            'plate': plate,
            'confidence': confidence,
            'timestamp': datetime.now()
        })

        # Need at least 3 readings for aggregation
        if len(self.plate_history[vehicle_id]) < 3:
            return plate, confidence

        # Extract plates and confidences
        readings = list(self.plate_history[vehicle_id])
        plates = [r['plate'] for r in readings]
        confidences = [r['confidence'] for r in readings]

        # Majority voting with confidence weighting
        plate_scores = {}
        for p, conf in zip(plates, confidences):
            plate_scores[p] = plate_scores.get(p, 0) + conf

        # Select plate with highest weighted score
        best_plate = max(plate_scores, key=plate_scores.get)

        # Calculate aggregated confidence (average of all readings of best_plate)
        best_confidences = [conf for p, conf in zip(plates, confidences) if p == best_plate]
        aggregated_conf = sum(best_confidences) / len(best_confidences)

        # Boost confidence if majority agrees
        agreement_ratio = plates.count(best_plate) / len(plates)
        if agreement_ratio >= 0.6:  # 60% agreement
            aggregated_conf = min(1.0, aggregated_conf * 1.2)  # 20% boost

        return best_plate, aggregated_conf

    def recognize_plate(self, vehicle_crop, min_confidence=0.5):
        """
        Recognize license plate number from vehicle crop.

        Args:
            vehicle_crop: Cropped image of detected vehicle (BGR)
            min_confidence: Minimum confidence threshold for OCR results

        Returns:
            Tuple (plate_number, confidence, plate_bbox)
            - plate_number: Cleaned plate text or None
            - confidence: OCR confidence score (0-1)
            - plate_bbox: Bounding box (x, y, w, h) or None
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, 0.0, None

        # Skip very small crops
        if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
            return None, 0.0, None

        # Detect potential plate regions
        plate_regions = self.detect_plate_region(vehicle_crop)

        best_plate = None
        best_confidence = 0.0
        best_bbox = None

        # Try OCR on detected plate regions first
        for (x, y, w, h) in plate_regions:
            # Extract plate region with some padding
            pad = 5
            y1 = max(0, y - pad)
            y2 = min(vehicle_crop.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(vehicle_crop.shape[1], x + w + pad)

            plate_roi = vehicle_crop[y1:y2, x1:x2]

            if plate_roi.size == 0:
                continue

            # Try both preprocessing methods (Indian plate specialized)
            binary_otsu, binary_adaptive = self.preprocess_indian_plate(plate_roi)

            # Perform OCR on both preprocessed images
            for preprocessed_img in [binary_otsu, binary_adaptive]:
                try:
                    results = self.reader.readtext(preprocessed_img, detail=1)

                    for (bbox, text, conf) in results:
                        cleaned_text = self.clean_plate_text(text)

                        if cleaned_text and conf > best_confidence:
                            best_plate = cleaned_text
                            best_confidence = conf
                            best_bbox = (x, y, w, h)
                except Exception:
                    # Silently handle OCR errors
                    continue

        # If no good plate found in detected regions, try whole vehicle crop
        if best_plate is None or best_confidence < min_confidence:
            try:
                # Try OCR on the entire vehicle crop (lower half)
                height = vehicle_crop.shape[0]
                lower_half = vehicle_crop[int(height * 0.4):, :]

                if lower_half.size > 0:
                    # Try both preprocessing methods
                    binary_otsu, binary_adaptive = self.preprocess_indian_plate(lower_half)

                    for preprocessed in [binary_otsu, binary_adaptive]:
                        results = self.reader.readtext(preprocessed, detail=1)

                        for (bbox, text, conf) in results:
                            cleaned_text = self.clean_plate_text(text)

                            if cleaned_text and conf > best_confidence:
                                best_plate = cleaned_text
                                best_confidence = conf
                                best_bbox = None  # No specific bbox for full image scan
            except Exception:
                # Silently handle OCR errors
                pass

        # Apply Indian format validation and correction
        if best_plate and best_confidence >= min_confidence:
            validated_plate, is_valid = self.validate_indian_plate_format(best_plate)

            if is_valid:
                if validated_plate != best_plate:
                    print(f"[LPR] Validated Indian plate: {validated_plate} (original: {best_plate})")
                return validated_plate, best_confidence, best_bbox
            else:
                # If strict validation is disabled, accept the plate anyway with a warning
                if not self.strict_validation:
                    print(f"[LPR] Warning: Non-standard format accepted: {best_plate} (confidence: {best_confidence:.2f})")
                    return best_plate, best_confidence, best_bbox
                else:
                    print(f"[LPR] Invalid format rejected: {best_plate}")
                    return None, 0.0, None

        # Log if plate was found but below confidence threshold
        if best_plate and best_confidence < min_confidence:
            print(f"[LPR] Plate found but confidence too low: {best_plate} (conf: {best_confidence:.2f}, min: {min_confidence})")

        return None, 0.0, None

    def batch_recognize(self, vehicle_crops, min_confidence=0.5):
        """
        Recognize license plates from multiple vehicle crops (batch processing).

        Args:
            vehicle_crops: List of vehicle crop images
            min_confidence: Minimum confidence threshold

        Returns:
            List of tuples (plate_number, confidence, plate_bbox) for each crop
        """
        results = []
        for crop in vehicle_crops:
            plate_info = self.recognize_plate(crop, min_confidence)
            results.append(plate_info)
        return results


def test_license_plate_recognizer():
    """Test function to demonstrate usage"""
    import os

    print("Testing License Plate Recognizer...")
    print("=" * 60)

    # Initialize recognizer
    recognizer = LicensePlateRecognizer()

    # Test with a sample image (if available)
    test_image_path = 'data/test_vehicle.jpg'

    if os.path.exists(test_image_path):
        # Load test image
        img = cv2.imread(test_image_path)

        # Recognize plate
        plate_number, confidence, bbox = recognizer.recognize_plate(img)

        if plate_number:
            print(f"Detected Plate: {plate_number}")
            print(f"Confidence: {confidence:.2%}")

            if bbox:
                print(f"Bounding Box: {bbox}")
                # Draw bbox on image
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, plate_number, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display result
            cv2.imshow('License Plate Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No license plate detected")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Recognizer initialized successfully. Ready for use.")


if __name__ == "__main__":
    test_license_plate_recognizer()

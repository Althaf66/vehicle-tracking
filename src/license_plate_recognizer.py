import cv2
import numpy as np
import easyocr
import torch
import re


class LicensePlateRecognizer:
    """
    License plate detection and recognition system using EasyOCR.
    Designed specifically for Camera 1 (entrance camera) only.
    """

    def __init__(self, gpu_enabled=None):
        """
        Initialize the license plate recognizer.

        Args:
            gpu_enabled: If None, auto-detect GPU availability. Otherwise, use the provided boolean.
        """
        if gpu_enabled is None:
            gpu_enabled = torch.cuda.is_available()

        # Initialize EasyOCR reader for license plate recognition
        # Using English language model for alphanumeric plates
        self.reader = easyocr.Reader(['en'], gpu=gpu_enabled)

        print(f"License Plate Recognizer initialized (GPU: {gpu_enabled})")

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

            # Preprocess the plate region
            preprocessed = self.preprocess_plate_region(plate_roi)

            # Perform OCR on preprocessed image
            try:
                results = self.reader.readtext(preprocessed, detail=1)

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
                    preprocessed = self.preprocess_plate_region(lower_half)
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

        # Return result only if confidence is above threshold
        if best_plate and best_confidence >= min_confidence:
            return best_plate, best_confidence, best_bbox

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

import cv2
import numpy as np
from datetime import datetime
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.multi_camera_tracker import VehicleClassifier
from src.license_plate_recognizer import LicensePlateRecognizer
from src.generate_id import VehicleIDGenerator
from ultralytics import YOLO
from backend.database import Database
from backend.config import config


class VideoProcessor:
    """
    Video processor for vehicle tracking with license plate recognition.
    Handles different processing logic for entrance camera (with LPR) vs monitoring cameras.
    """

    def __init__(self):
        """Initialize video processor with ML models and database"""
        print("Initializing Video Processor...")

        # Initialize database
        self.db = Database(config.DATABASE_URL)

        # Initialize ML models
        self.classifier = VehicleClassifier(
            config.COLOR_MODEL_PATH,
            config.CARNAME_MODEL_PATH
        )

        self.vehicle_detector = YOLO(config.YOLO_MODEL_PATH)

        # License plate recognizer (for Camera 1 only)
        self.plate_recognizer = LicensePlateRecognizer(
            plate_model_path=config.PLATE_DETECTOR_MODEL_PATH,
            plate_detection_confidence=config.PLATE_DETECTION_CONFIDENCE,
            strict_validation=config.STRICT_PLATE_VALIDATION
        )

        # ID generator for vehicle identification
        self.id_generator = VehicleIDGenerator(
            color_confidence_threshold=0.6,
            carname_confidence_threshold=0.7
        )

        # In-memory cache for recent unauthorized vehicles (for matching across cameras)
        # Format: {tracking_id: {'color': str, 'car_name': str, 'last_seen': datetime}}
        self.unauthorized_cache = {}

        print("Video Processor initialized successfully")

    def generate_vehicle_id(self, color: str, car_name: str, color_conf: float, carname_conf: float) -> tuple:
        """
        Generate unique vehicle ID using VehicleIDGenerator.
        Returns: (vehicle_id, is_valid)
        """
        vehicle_id, is_valid = self.id_generator.generate_id(color, car_name, color_conf, carname_conf)
        return vehicle_id, is_valid

    def process_camera1_entrance(self, frame, frame_number):
        """
        Process entrance camera (Camera 1) with license plate recognition and authorization check.

        Returns:
            detections: List of detection dictionaries with authorization status
        """
        # Detect vehicles
        results = self.vehicle_detector(
            frame,
            classes=[2, 5, 7],  # car, bus, truck
            conf=config.DETECTION_CONFIDENCE,
            verbose=False
        )

        detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)

                # Crop vehicle
                vehicle_crop = frame[y1:y2, x1:x2]

                # Skip small detections
                if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
                    continue

                # Classify vehicle (color and car name)
                color, car_name, color_conf, carname_conf = self.classifier.predict(vehicle_crop)

                # Calculate average confidence score (as percentage 0-100)
                confidence_score = ((color_conf + carname_conf) / 2) * 100

                # Generate vehicle ID using VehicleIDGenerator
                vehicle_id, is_valid = self.generate_vehicle_id(color, car_name, color_conf, carname_conf)

                if not is_valid:
                    # Skip if confidence is too low
                    continue

                # License plate recognition (skip frames for performance)
                plate_number, plate_conf, plate_bbox = None, 0.0, None
                if frame_number % config.LPR_FRAME_SKIP == 0:
                    plate_number, plate_conf, plate_bbox = self.plate_recognizer.recognize_plate(
                        vehicle_crop, min_confidence=config.PLATE_RECOGNITION_CONFIDENCE
                    )

                # Apply multi-frame aggregation if plate detected
                if plate_number and vehicle_id:
                    plate_number, plate_conf = self.plate_recognizer.aggregate_plate_readings(
                        str(vehicle_id), plate_number, plate_conf
                    )
                    print(f"[CAMERA1] Aggregated plate: {plate_number} (conf: {plate_conf:.2f})")

                # Log plate detection
                if plate_number:
                    is_authorized_check = self.db.is_vehicle_authorized(plate_number)
                    print(f"[CAMERA1-LPR] Plate: {plate_number} | ID: {vehicle_id} | "
                          f"Vehicle: {color} {car_name} | Conf: {confidence_score:.1f}% | Authorized: {is_authorized_check}")

                # Check authorization and create tracking record
                is_authorized = False
                unique_id = str(vehicle_id)  # Convert to string for database storage

                if plate_number:
                    is_authorized = self.db.is_vehicle_authorized(plate_number)

                    # If unauthorized, create tracking record
                    if not is_authorized:
                        # Check if tracking record exists
                        existing = self.db.get_unauthorized_tracking_by_id(unique_id)
                        if not existing:
                            # Create new tracking record
                            self.db.add_unauthorized_tracking(
                                tracking_id=unique_id,
                                plate_number=plate_number,
                                color=color,
                                car_name=car_name,
                                confidence_score=confidence_score,
                                plate_confidence=plate_conf,
                                camera_id='camera1'
                            )
                            print(f"[CAMERA1] Created tracking record | ID: {unique_id} | Plate: {plate_number} (conf: {plate_conf:.2f})")
                        else:
                            # Update existing record with camera1 sighting
                            self.db.update_unauthorized_tracking(
                                tracking_id=unique_id,
                                camera_id='camera1'
                            )
                            # Update plate if this reading has higher confidence
                            if self.db.update_plate_if_better(unique_id, plate_number, plate_conf):
                                print(f"[CAMERA1] Updated plate (better confidence) | ID: {unique_id} | Plate: {plate_number} (conf: {plate_conf:.2f})")
                            else:
                                print(f"[CAMERA1] Kept existing plate (higher confidence) | ID: {unique_id}")

                        # Add/update cache
                        self.unauthorized_cache[unique_id] = {
                            'color': color,
                            'car_name': car_name,
                            'plate': plate_number,
                            'confidence_score': confidence_score,
                            'last_seen': datetime.now()
                        }

                # If no plate detected, create tracking record for the vehicle
                elif color and car_name:
                    # Check if tracking record exists
                    existing = self.db.get_unauthorized_tracking_by_id(unique_id)
                    if not existing:
                        # Create new tracking record without plate
                        self.db.add_unauthorized_tracking(
                            tracking_id=unique_id,
                            plate_number=None,
                            color=color,
                            car_name=car_name,
                            confidence_score=confidence_score,
                            camera_id='camera1'
                        )
                        print(f"[CAMERA1] Created tracking record (no plate) | ID: {unique_id}")
                    else:
                        # Update existing record
                        self.db.update_unauthorized_tracking(
                            tracking_id=unique_id,
                            camera_id='camera1'
                        )

                    # Add/update cache
                    self.unauthorized_cache[unique_id] = {
                        'color': color,
                        'car_name': car_name,
                        'plate': None,
                        'confidence_score': confidence_score,
                        'last_seen': datetime.now()
                    }

                detections.append({
                    'bbox': bbox,
                    'color': color,
                    'car_name': car_name,
                    'color_conf': color_conf,
                    'carname_conf': carname_conf,
                    'confidence_score': confidence_score,
                    'unique_id': unique_id,
                    'license_plate': plate_number,
                    'plate_confidence': plate_conf,
                    'plate_bbox': plate_bbox,
                    'is_authorized': is_authorized,
                    'camera_id': 'camera1'
                })

        return detections

    def process_monitoring_camera(self, frame, camera_id, frame_number):
        """
        Process monitoring cameras (Camera 2, 3, 4) - no LPR, but match with existing tracking.

        Returns:
            detections: List of detection dictionaries with tracking info
        """
        # Detect vehicles
        results = self.vehicle_detector(
            frame,
            classes=[2, 5, 7],  # car, bus, truck
            conf=config.DETECTION_CONFIDENCE,
            verbose=False
        )

        detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)

                # Crop vehicle
                vehicle_crop = frame[y1:y2, x1:x2]

                # Skip small detections
                if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
                    continue

                # Classify vehicle (color and car name)
                color, car_name, color_conf, carname_conf = self.classifier.predict(vehicle_crop)

                # Calculate average confidence score (as percentage 0-100)
                confidence_score = ((color_conf + carname_conf) / 2) * 100

                # Generate vehicle ID using VehicleIDGenerator
                vehicle_id, is_valid = self.generate_vehicle_id(color, car_name, color_conf, carname_conf)

                if not is_valid:
                    # Skip if confidence is too low
                    continue

                unique_id = str(vehicle_id)  # Convert to string for database storage

                # Check if tracking record exists
                existing = self.db.get_unauthorized_tracking_by_id(unique_id)

                if existing:
                    # Update tracking record with new camera sighting
                    self.db.update_unauthorized_tracking(
                        tracking_id=unique_id,
                        camera_id=camera_id
                    )
                    # print(f"[{camera_id.upper()}] Updated tracking record | ID: {unique_id}")
                else:
                    # Create new tracking record for this vehicle
                    self.db.add_unauthorized_tracking(
                        tracking_id=unique_id,
                        plate_number=None,  # Monitoring cameras don't have LPR
                        color=color,
                        car_name=car_name,
                        confidence_score=confidence_score,
                        camera_id=camera_id
                    )
                    print(f"[{camera_id.upper()}] Created tracking record | ID: {unique_id}")

                # Update cache
                self.unauthorized_cache[unique_id] = {
                    'color': color,
                    'car_name': car_name,
                    'plate': existing.get('plate_number') if existing else None,
                    'confidence_score': confidence_score,
                    'last_seen': datetime.now()
                }

                detections.append({
                    'bbox': bbox,
                    'color': color,
                    'car_name': car_name,
                    'color_conf': color_conf,
                    'carname_conf': carname_conf,
                    'confidence_score': confidence_score,
                    'unique_id': unique_id,
                    'camera_id': camera_id
                })

        return detections

    def draw_detections_entrance(self, frame, detections):
        """
        Draw detection boxes for entrance camera with authorization status.
        - Green: Authorized vehicle (with plate)
        - Red: Unauthorized vehicle (with plate)
        - Yellow: Vehicle detected but no plate visible
        Shows: Unique ID, Confidence, Vehicle details
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            plate = det.get('license_plate')
            is_authorized = det.get('is_authorized', False)
            unique_id = det.get('unique_id')
            confidence = det.get('confidence_score', 0)

            # Determine box color based on authorization status
            if plate:
                if is_authorized:
                    box_color = (0, 255, 0)  # Green for authorized
                    status_label = "AUTHORIZED"
                else:
                    box_color = (0, 0, 255)  # Red for unauthorized
                    status_label = "UNAUTHORIZED"
            else:
                box_color = (0, 255, 255)  # Yellow for no plate detected
                status_label = "NO PLATE"

            # Build label with unique ID and confidence
            if unique_id:
                # Shorten unique ID for display (first 12 chars)
                short_id = unique_id[:12] + "..." if len(unique_id) > 12 else unique_id
                label = f"{status_label} | ID: {short_id} | Conf: {confidence:.1f}%"
            else:
                label = f"{status_label} | {det['color']} {det['car_name']} | Conf: {confidence:.1f}%"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw label with background (multi-line if needed)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

            # Draw plate info below if available
            if plate:
                plate_label = f"Plate: {plate}"
                (plate_w, plate_h), _ = cv2.getTextSize(plate_label, font, font_scale, font_thickness)
                cv2.rectangle(frame, (x1, y1 - label_h - plate_h - 15), (x1 + plate_w + 10, y1 - label_h - 10), box_color, -1)
                cv2.putText(frame, plate_label, (x1 + 5, y1 - label_h - 12), font, font_scale, (255, 255, 255), font_thickness)

            # Draw plate bounding box if available
            if plate and det.get('plate_bbox'):
                px, py, pw, ph = det['plate_bbox']
                plate_x1 = x1 + px
                plate_y1 = y1 + py
                plate_x2 = plate_x1 + pw
                plate_y2 = plate_y1 + ph
                cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 255, 0), 2)

        return frame

    def draw_detections_monitoring(self, frame, detections):
        """
        Draw detection boxes for monitoring cameras.
        - Blue: Vehicle detected
        Shows: Unique ID, Confidence, Vehicle details
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            unique_id = det.get('unique_id')
            confidence = det.get('confidence_score', 0)

            box_color = (255, 0, 0)  # Blue for monitoring cameras

            # Build label with unique ID and confidence
            if unique_id:
                # Shorten unique ID for display
                short_id = unique_id[:12] + "..." if len(unique_id) > 12 else unique_id
                label = f"ID: {short_id} | Conf: {confidence:.1f}%"
                vehicle_label = f"{det['color']} {det['car_name']}"
            else:
                label = f"{det['color']} {det['car_name']} | Conf: {confidence:.1f}%"
                vehicle_label = None

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw labels with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw main label
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

            # Draw vehicle info below if we have unique_id
            if vehicle_label:
                (veh_w, veh_h), _ = cv2.getTextSize(vehicle_label, font, font_scale, font_thickness)
                cv2.rectangle(frame, (x1, y1 - label_h - veh_h - 15), (x1 + veh_w + 10, y1 - label_h - 10), box_color, -1)
                cv2.putText(frame, vehicle_label, (x1 + 5, y1 - label_h - 12), font, font_scale, (255, 255, 255), font_thickness)

        return frame


def stream_video(video_path: str, camera_id: str, camera_type: str):
    """
    Generator function for streaming video with vehicle detection.
    Yields MJPEG frames for HTTP streaming.

    Args:
        video_path: Path to video file
        camera_id: Camera identifier (camera1, camera2, etc.)
        camera_type: 'entrance' or 'monitoring'
    """
    # Initialize processor
    processor = VideoProcessor()

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_number = 0
                continue

            # Process frame based on camera type
            if camera_type == 'entrance':
                detections = processor.process_camera1_entrance(frame, frame_number)
                frame_with_detections = processor.draw_detections_entrance(frame.copy(), detections)

                # Log plate detections in stream
                plate_detections = [d for d in detections if d.get('license_plate')]
                if plate_detections:
                    for det in plate_detections:
                        print(f"[STREAM] Frame {frame_number} | {det['license_plate']} | "
                              f"Conf: {det['plate_confidence']:.2%}")
            else:
                detections = processor.process_monitoring_camera(frame, camera_id, frame_number)
                frame_with_detections = processor.draw_detections_monitoring(frame.copy(), detections)

            # Add camera info overlay
            camera_label = f"Camera {camera_id[-1]}"
            if camera_type == 'entrance':
                camera_label += " - Entrance (LPR)"
            else:
                camera_label += " - Monitoring"

            cv2.putText(frame_with_detections, camera_label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add detection count
            detection_count = len(detections)
            cv2.putText(frame_with_detections, f"Vehicles: {detection_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame_with_detections)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            frame_number += 1

    except GeneratorExit:
        # Client disconnected
        pass
    finally:
        cap.release()
        print(f"Stream closed for {camera_id}")


if __name__ == "__main__":
    # Test video processor
    print("Testing Video Processor...")
    processor = VideoProcessor()
    print("Video Processor test successful!")

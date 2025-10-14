import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import json
from collections import defaultdict, deque
from scipy import stats
import time
import os
import pandas as pd
from datetime import datetime
from generate_id import VehicleIDGenerator

class VehicleClassifier:
    def __init__(self, color_model_path, carname_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.color_model = self.load_model(color_model_path, 'models/color_classes.json')
        self.carname_model = self.load_model(carname_model_path, 'models/carname_classes.json')
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path, classes_path):
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        
        from torchvision import models
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(classes))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return {'model': model, 'classes': classes}
    
    def predict(self, image):
        """
        Predict color and car name with confidence scores
        Returns: (color, car_name, color_confidence, carname_confidence)
        """
        # Convert to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Predict color
            color_output = self.color_model['model'](image_tensor)
            color_probs = torch.softmax(color_output, dim=1)
            color_conf, color_pred = torch.max(color_probs, 1)
            color = self.color_model['classes'][color_pred.item()]
            color_confidence = color_conf.item()
            
            # Predict car name
            carname_output = self.carname_model['model'](image_tensor)
            carname_probs = torch.softmax(carname_output, dim=1)
            carname_conf, carname_pred = torch.max(carname_probs, 1)
            car_name = self.carname_model['classes'][carname_pred.item()]
            carname_confidence = carname_conf.item()
        
        return color, car_name, color_confidence, carname_confidence


class TemporalSmoother:
    """
    Maintains temporal consistency of vehicle IDs using a sliding window
    """
    def __init__(self, window_size=10, iou_threshold=0.5):
        self.window_size = window_size  # Number of frames to look back
        self.iou_threshold = iou_threshold
        # Store recent detections: {track_id: deque of (bbox, vehicle_id, confidence)}
        self.track_history = defaultdict(lambda: deque(maxlen=window_size))
        self.next_track_id = 0
    
    def compute_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection area
        intersect_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        intersect_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersect_area = intersect_w * intersect_h
        
        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersect_area
        
        if union_area == 0:
            return 0
        
        return intersect_area / union_area
    
    def match_detection_to_track(self, bbox, current_detections):
        """
        Match current detection to existing tracks using IoU
        Returns: track_id (existing or new)
        """
        best_iou = 0
        best_track_id = None
        
        # Compare with all existing tracks
        for track_id, history in self.track_history.items():
            if len(history) > 0:
                # Get most recent bbox from this track
                last_bbox, _, _ = history[-1]
                iou = self.compute_iou(bbox, last_bbox)
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
        
        # If no match found, create new track
        if best_track_id is None:
            best_track_id = self.next_track_id
            self.next_track_id += 1
        
        return best_track_id
    
    def get_smoothed_id(self, track_id, current_vehicle_id, current_conf, bbox):
        """
        Apply temporal smoothing to get stable vehicle ID
        Uses majority voting over the sliding window
        """
        # Add current detection to track history
        self.track_history[track_id].append((bbox, current_vehicle_id, current_conf))
        
        # Get all vehicle IDs from history (excluding None)
        history = self.track_history[track_id]
        vehicle_ids = [vid for _, vid, _ in history if vid is not None]
        
        if len(vehicle_ids) == 0:
            return None
        
        # Use majority voting (mode)
        if len(vehicle_ids) >= 3:  # Need at least 3 samples for smoothing
            mode_result = stats.mode(vehicle_ids, keepdims=True)
            smoothed_id = int(mode_result.mode[0])
            
            # Calculate stability score (how many frames agree)
            stability = list(vehicle_ids).count(smoothed_id) / len(vehicle_ids)
            
            # Only return smoothed ID if stability is high enough
            if stability >= 0.6:  # At least 60% agreement
                return smoothed_id
        
        # If not enough data or low stability, return current ID
        return current_vehicle_id
    
    def cleanup_old_tracks(self, current_frame, max_age=30):
        """Remove tracks that haven't been updated recently"""
        tracks_to_remove = []
        for track_id, history in self.track_history.items():
            if len(history) == 0:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]


class MultiCameraTracker:
    def __init__(self, classifier, id_generator, 
                 detection_conf=0.4,
                 temporal_window=10,
                 iou_threshold=0.5):
        self.classifier = classifier
        self.id_generator = id_generator
        self.vehicle_detector = YOLO('yolov8n.pt')
        
        # Temporal smoothers for each camera
        self.smoothers = {}
        self.temporal_window = temporal_window
        self.iou_threshold = iou_threshold
        self.detection_conf = detection_conf
        
        # Track vehicles across cameras
        self.global_tracks = defaultdict(lambda: {
            'color': None,
            'car_name': None,
            'last_seen': {},
            'cameras': set(),
            'first_seen': None,
            'confidence_history': []
        })
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'accepted_detections': 0,
            'rejected_low_confidence': 0
        }
    
    def get_or_create_smoother(self, camera_id):
        """Get smoother for a camera, create if doesn't exist"""
        if camera_id not in self.smoothers:
            self.smoothers[camera_id] = TemporalSmoother(
                window_size=self.temporal_window,
                iou_threshold=self.iou_threshold
            )
        return self.smoothers[camera_id]
    
    def process_frame(self, frame, camera_id, frame_number):
        """Process single frame with temporal smoothing and confidence filtering"""
        # Detect vehicles
        results = self.vehicle_detector(
            frame, 
            classes=[2, 5, 7],  # car, bus, truck
            conf=self.detection_conf,
            verbose=False
        )
        
        detections = []
        smoother = self.get_or_create_smoother(camera_id)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                self.stats['total_detections'] += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)
                detection_conf = float(box.conf[0])
                
                # Crop vehicle
                vehicle_crop = frame[y1:y2, x1:x2]
                
                # Skip small detections
                if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
                    continue
                
                # Classify with confidence
                color, car_name, color_conf, carname_conf = self.classifier.predict(vehicle_crop)
                
                # Generate ID with confidence check
                vehicle_id, is_valid = self.id_generator.generate_id(
                    color, car_name, color_conf, carname_conf
                )
                
                if not is_valid:
                    # Reject low confidence prediction
                    self.stats['rejected_low_confidence'] += 1
                    detections.append({
                        'id': None,
                        'bbox': bbox,
                        'color': color,
                        'car_name': car_name,
                        'confidence': min(color_conf, carname_conf),
                        'status': 'LOW_CONFIDENCE'
                    })
                    continue
                
                self.stats['accepted_detections'] += 1
                
                # Match to track and apply temporal smoothing
                track_id = smoother.match_detection_to_track(bbox, detections)
                smoothed_id = smoother.get_smoothed_id(
                    track_id, vehicle_id, 
                    min(color_conf, carname_conf), bbox
                )
                
                # Use smoothed ID if available
                final_id = smoothed_id if smoothed_id is not None else vehicle_id
                
                # Update global tracking
                if final_id is not None:
                    if self.global_tracks[final_id]['first_seen'] is None:
                        self.global_tracks[final_id]['first_seen'] = frame_number
                    
                    self.global_tracks[final_id]['color'] = color
                    self.global_tracks[final_id]['car_name'] = car_name
                    self.global_tracks[final_id]['last_seen'][camera_id] = frame_number
                    self.global_tracks[final_id]['cameras'].add(camera_id)
                    self.global_tracks[final_id]['confidence_history'].append(
                        min(color_conf, carname_conf)
                    )
                
                detections.append({
                    'id': final_id,
                    'track_id': track_id,
                    'bbox': bbox,
                    'color': color,
                    'car_name': car_name,
                    'color_conf': color_conf,
                    'carname_conf': carname_conf,
                    'confidence': min(color_conf, carname_conf),
                    'status': 'TRACKED' if smoothed_id else 'NEW'
                })
        
        # Cleanup old tracks
        smoother.cleanup_old_tracks(frame_number)
        
        return detections
    
    def draw_detections(self, frame, detections, show_confidence=True):
        """Draw bounding boxes and IDs with confidence information"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            vehicle_id = det['id']
            status = det.get('status', 'TRACKED')
            
            # Color based on status
            if status == 'LOW_CONFIDENCE':
                box_color = (0, 0, 255)  # Red for low confidence
                label = f"LOW CONF: {det['color']} {det['car_name']}"
            elif status == 'NEW':
                box_color = (0, 165, 255)  # Orange for new
                label = f"ID:{vehicle_id} {det['color']} {det['car_name']}"
            else:
                box_color = (0, 255, 0)  # Green for tracked
                label = f"ID:{vehicle_id} {det['color']} {det['car_name']}"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Add confidence if requested
            if show_confidence and 'confidence' in det:
                label += f" ({det['confidence']:.2f})"
            
            # Draw label with background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w, y1), box_color, -1)
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def get_statistics(self):
        """Get tracking statistics"""
        return {
            **self.stats,
            'acceptance_rate': self.stats['accepted_detections'] / max(1, self.stats['total_detections']),
            'unique_vehicles': len(self.global_tracks)
        }


def main():
    print("Initializing Multi-Camera Vehicle Tracking System...")
    print("=" * 60)
    
    # Initialize components
    classifier = VehicleClassifier(
        'models/color_classifier.pth',
        'models/car_name_classifier.pth'
    )
    
    # Configure thresholds
    id_generator = VehicleIDGenerator(
        color_confidence_threshold=0.6,    # Adjust as needed
        carname_confidence_threshold=0.7   # Adjust as needed
    )
    
    tracker = MultiCameraTracker(
        classifier, 
        id_generator,
        detection_conf=0.4,      # YOLO detection confidence
        temporal_window=10,      # Frames for temporal smoothing
        iou_threshold=0.5        # IoU threshold for matching
    )
    
    # Open videos
    videos = {
        'entrance_cam': cv2.VideoCapture('data/raw_videos/a1.mp4'),
        'ec_department_cam': cv2.VideoCapture('data/raw_videos/a2.mp4')
    }

    # Create output directory
    os.makedirs('output/tracked_videos', exist_ok=True)

    # Video writers for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = {}
    for cam_id, cap in videos.items():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        writers[cam_id] = cv2.VideoWriter(
            f'output/tracked_videos/tracked_a{cam_id}.mp4',
            fourcc, fps, (width, height)
        )
    
    frame_number = 0
    start_time = time.time()
    
    print("\nProcessing videos...")
    print("Processing started. Please wait...\n")
    
    while True:
        frames = {}
        
        # Read frames from all cameras
        for cam_id, cap in videos.items():
            ret, frame = cap.read()
            if not ret:
                break
            frames[cam_id] = frame
        
        if len(frames) != len(videos):
            break
        
        # Process each camera
        for cam_id, frame in frames.items():
            detections = tracker.process_frame(frame, cam_id, frame_number)
            frame_with_detections = tracker.draw_detections(
                frame.copy(), detections, show_confidence=True
            )
            
            # Add frame info
            info_text = f"Frame: {frame_number} | Camera: {cam_id} | Vehicles: {len([d for d in detections if d['id'] is not None])}"
            cv2.putText(frame_with_detections, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save video frame
            writers[cam_id].write(frame_with_detections)
        
        frame_number += 1
        
        # Progress update every 100 frames
        if frame_number % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_number / elapsed
            print(f"Frame {frame_number} | FPS: {fps_processing:.2f} | "
                  f"Unique vehicles: {len(tracker.global_tracks)}")
    
    # Cleanup
    for cap in videos.values():
        cap.release()
    for writer in writers.values():
        writer.release()
    
    print("\nProcessing completed. Videos saved in 'output/tracked_videos' directory.")
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("TRACKING SUMMARY")
    print("=" * 60)
    
    stats = tracker.get_statistics()
    print(f"\nDetection Statistics:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Accepted (high confidence): {stats['accepted_detections']}")
    print(f"  Rejected (low confidence): {stats['rejected_low_confidence']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"  Unique vehicles tracked: {stats['unique_vehicles']}")
    
    print(f"\nVehicle Details:")
    
    # Prepare data for Excel
    excel_data = []
    for vehicle_id, info in sorted(tracker.global_tracks.items()):
        avg_conf = np.mean(info['confidence_history']) if info['confidence_history'] else 0
        
        # Print to console
        print(f"\n  ID {vehicle_id}: {info['color']} {info['car_name']}")
        print(f"    Cameras: {sorted(info['cameras'])}")
        print(f"    First seen: Frame {info['first_seen']}")
        print(f"    Last seen: {info['last_seen']}")
        print(f"    Avg confidence: {avg_conf:.2f}")
        
        # Prepare row for Excel
        excel_row = {
            'Vehicle ID': vehicle_id,
            'Color': info['color'],
            'Car Name': info['car_name'],
            'Cameras': ', '.join(map(str, sorted(info['cameras']))),
            'First Seen Frame': info['first_seen'],
            'Average Confidence': f"{avg_conf:.2f}",
        }
        
        # Add last seen frames for each camera
        for cam_id, frame in info['last_seen'].items():
            excel_row[f'Last Seen (Camera {cam_id})'] = frame
        
        excel_data.append(excel_row)
    
    if excel_data:
        # Create output directory if it doesn't exist
        os.makedirs('output/reports', exist_ok=True)
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(excel_data)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = f'output/reports/vehicle_tracking_report_{timestamp}.xlsx'
        
        # Save to Excel
        df.to_excel(excel_path, index=False, sheet_name='Vehicle Tracking')
        print(f"\nVehicle tracking report saved to: {excel_path}")

if __name__ == "__main__":
    main()
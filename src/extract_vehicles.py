import cv2
from ultralytics import YOLO
import os

def extract_vehicles_from_video(video_path, output_dir):
    """Extract vehicle images from video"""
    
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    vehicle_count = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles every 10 frames to avoid duplicates
        if frame_count % 10 == 0:
            results = model(frame, classes=[2, 5, 7])  # car, bus, truck
            
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                
                # Crop vehicle
                vehicle_crop = frame[y1:y2, x1:x2]
                
                # Save if crop is valid size
                if vehicle_crop.shape[0] > 50 and vehicle_crop.shape[1] > 50:
                    filename = f"vehicle_n2{vehicle_count:05d}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), vehicle_crop)
                    vehicle_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {vehicle_count} vehicle images")

extract_vehicles_from_video('data/raw_videos/camera_ec_new.mp4', 'data/extracted_vehicles/n2/')
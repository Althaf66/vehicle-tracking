import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""

    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
    ALGORITHM = os.getenv('ALGORITHM', 'HS256')
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '1440'))  # 24 hours

    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'vehicle_tracking.db')

    # CORS (allowed origins for frontend)
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000').split(',')

    # Video files configuration
    VIDEO_DIR = os.getenv('VIDEO_DIR', 'src/data/raw_videos')
    CAMERA_VIDEOS = {
        'camera1':'rtsp://admin:1234qwer@192.168.98.16:554/cam/realmonitor?channel=1&subtype=1', 
        'camera2': os.path.join(VIDEO_DIR, 'ec.mp4'),
        'camera3': os.path.join(VIDEO_DIR, 'sports.mp4'),
        'camera4': os.path.join(VIDEO_DIR, 'exit.mp4'),
    }

    # ML Models paths
    MODELS_DIR = 'src/models'
    COLOR_MODEL_PATH = os.path.join(MODELS_DIR, 'color_classifier.pth')
    CARNAME_MODEL_PATH = os.path.join(MODELS_DIR, 'car_name_classifier.pth')
    YOLO_MODEL_PATH = 'yolov8n.pt'

    # Video processing settings
    DETECTION_CONFIDENCE = float(os.getenv('DETECTION_CONFIDENCE', '0.25'))
    PLATE_RECOGNITION_CONFIDENCE = float(os.getenv('PLATE_RECOGNITION_CONFIDENCE', '0.20'))
    TEMPORAL_WINDOW = int(os.getenv('TEMPORAL_WINDOW', '10'))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.5'))

    # Tracking settings
    VEHICLE_MATCH_TIME_WINDOW = int(os.getenv('VEHICLE_MATCH_TIME_WINDOW', '5'))  # minutes

    # LPR Improvements for Indian Plates
    INDIAN_PLATE_FORMAT_VALIDATION = os.getenv('INDIAN_PLATE_FORMAT_VALIDATION', 'true').lower() == 'true'
    PLATE_AGGREGATION_WINDOW = int(os.getenv('PLATE_AGGREGATION_WINDOW', '25'))
    CHARACTER_CONFUSION_CORRECTION = os.getenv('CHARACTER_CONFUSION_CORRECTION', 'true').lower() == 'true'
    PLATE_VALIDATION_MIN_SCORE = float(os.getenv('PLATE_VALIDATION_MIN_SCORE', '0.60'))
    STRICT_PLATE_VALIDATION = os.getenv('STRICT_PLATE_VALIDATION', 'false').lower() == 'true'

    # YOLO Plate Detection + Ollama OCR Settings
    PLATE_DETECTOR_MODEL_PATH = os.getenv('PLATE_DETECTOR_MODEL_PATH', os.path.join(MODELS_DIR, 'plate_detector', 'best.pt'))
    PLATE_DETECTION_CONFIDENCE = float(os.getenv('PLATE_DETECTION_CONFIDENCE', '0.35'))
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_OCR_MODEL = os.getenv('OLLAMA_OCR_MODEL', 'glm-ocr')
    LPR_REQUEST_TIMEOUT = int(os.getenv('LPR_REQUEST_TIMEOUT', '30'))
    LPR_FRAME_SKIP = int(os.getenv('LPR_FRAME_SKIP', '1'))


config = Config()

command to run
```
cd frontend
npm install
```

Place 4 test video files in `src/data/raw_videos/`:
- `entrance.mp4` - Entrance camera (vehicles with visible license plates)
- `ec.mp4` - Monitoring camera
- `sports.mp4` - Monitoring camera
- `exit.mp4` - Monitoring camera

```
vehicle_tracking\Scripts\activate
pip install -r requirements.txt
python run_backend.py
```

# Vehicle Tracking System with License Plate Recognition

A comprehensive full-stack web application for multi-camera vehicle tracking with automated license plate recognition, authorization checking, and real-time monitoring.

## Features

### Core Capabilities
- **Multi-Camera Vehicle Detection**: Track vehicles across 4 cameras simultaneously
- **License Plate Recognition**: Automatic LPR on entrance camera (Camera 1)
- **Authorization System**: Automatic vehicle authorization checking
- **Unauthorized Vehicle Tracking**: Track unauthorized vehicles across all cameras
- **Real-time Video Streaming**: Live MJPEG video streams with detection overlays
- **User Authentication**: JWT-based authentication with role-based access control
- **Admin Dashboard**: Manage authorized vehicles (CRUD operations)

### Technical Features
- YOLOv8n vehicle detection
- ResNet-18 color classification (8 colors)
- ResNet-18 car model classification (73 models)
- EasyOCR license plate recognition
- Temporal smoothing for stable tracking
- Multi-camera vehicle correlation
- SQLite database for data persistence
- RESTful API backend with FastAPI
- Modern React frontend with real-time updates

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     React Frontend                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Home: 4 Camera Feeds + Tracking Table             │ │
│  │  Authorized Vehicles Management (Admin)            │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP/REST API (JWT Auth)
┌──────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                          │
│  ┌─────────────┐  ┌────────────────────────────────────┐│
│  │ Auth System │  │     Video Processing               ││
│  │ JWT Tokens  │  │  • YOLOv8 Detection                ││
│  └─────────────┘  │  • LPR (Camera 1)                  ││
│  ┌─────────────┐  │  • Color/Model Classification      ││
│  │   Database  │  │  • Authorization Check             ││
│  │   SQLite    │  │  • Tracking Database Updates       ││
│  └─────────────┘  └────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
vehicle_tracking_project/
├── backend/                    # FastAPI backend
│   ├── main.py                # API endpoints
│   ├── database.py            # Database operations
│   ├── auth.py                # Authentication
│   ├── models.py              # Pydantic models
│   ├── config.py              # Configuration
│   ├── video_processor.py     # Video streaming & processing
│   ├── init_db.py            # Database initialization
│   └── .env                   # Environment variables
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Page components
│   │   ├── contexts/         # React contexts
│   │   ├── services/         # API services
│   │   └── App.jsx           # Main app
│   └── package.json
├── src/                       # ML pipeline
│   ├── multi_camera_tracker.py
│   ├── license_plate_recognizer.py
│   ├── generate_id.py
│   └── data/
│       └── raw_videos/       # Video files location
├── models/                    # Trained ML models
│   ├── yolov8n.pt
│   ├── color_classifier.pth
│   ├── car_name_classifier.pth
│   ├── color_classes.json
│   └── carname_classes.json
├── requirements.txt           # Python dependencies
├── run_backend.py            # Backend startup script
└── README.md                 # This file
```

## Installation & Setup

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- 4GB+ RAM (for ML models)
- GPU recommended (CUDA-capable) but CPU works

### Step 1: Clone and Setup Python Environment

```bash
cd vehicle_tracking_project

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Video Files

Place 4 test video files in `src/data/raw_videos/`:
- `camera1.mp4` - Entrance camera (vehicles with visible license plates)
- `camera2.mp4` - Monitoring camera
- `camera3.mp4` - Monitoring camera
- `camera4.mp4` - Monitoring camera

**Note**: Ensure `camera1.mp4` shows vehicles entering with clearly visible license plates for best LPR results.

### Step 3: Initialize Backend Database

```bash
python backend/init_db.py
```

This creates:
- SQLite database with required tables
- Default admin user (username: `admin`, password: `admin123`)

### Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Step 5: Configure Environment (Optional)

Edit `backend/.env` to customize:
```env
SECRET_KEY=your-secret-key-here
DATABASE_URL=vehicle_tracking.db
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Running the Application

### Terminal 1: Start Backend Server

```bash
python run_backend.py
```

Backend will start at: http://localhost:8001
- API documentation: http://localhost:8001/docs
- Alternative docs: http://localhost:8001/redoc

### Terminal 2: Start Frontend Development Server

```bash
cd frontend
npm run dev
```

Frontend will start at: http://localhost:5173

## Usage Guide

### First-Time Login

1. Open http://localhost:5173
2. Click "Login"
3. Use default admin credentials:
   - Username: `admin`
   - Password: `admin123`

### Managing Authorized Vehicles (Admin)

1. Navigate to "Authorized Vehicles" in the navbar
2. Click "+ Add Vehicle"
3. Enter vehicle details:
   - Plate number (required)
   - Owner name
   - Vehicle type
   - Notes
4. Submit to authorize the vehicle

### Monitoring Dashboard

The home page displays:

**Top Section - 4 Camera Feeds (2x2 Grid)**:
- **Camera 1 (Top Left)** - Entrance with LPR
  - 🟢 Green box: Authorized vehicle (with plate number)
  - 🔴 Red box: Unauthorized vehicle (with plate number)
  - 🟡 Yellow box: Vehicle detected but no plate visible
- **Cameras 2, 3, 4** - Monitoring only
  - 🔵 Blue box: Vehicle detected
  - Shows color and model if classified
  - Displays tracking ID if matched to unauthorized vehicle

**Bottom Section - Unauthorized Vehicle Tracking Table**:
- Real-time tracking of unauthorized vehicles
- Shows: Tracking ID, Plate, Color, Model, Camera Path, Timestamps, Status
- Auto-refreshes every 5 seconds
- Toggle to show active vehicles only

## How It Works

### Camera 1 (Entrance - with LPR)

1. Detects vehicle entering (YOLOv8)
2. Recognizes license plate (EasyOCR)
3. Checks plate against authorized database
4. **If AUTHORIZED**: Shows green box, allows entry
5. **If UNAUTHORIZED**:
   - Shows red box
   - Creates tracking record in database
   - Assigns unique tracking ID
   - Records timestamp and camera

### Cameras 2, 3, 4 (Monitoring - No LPR)

1. Detects vehicles (YOLOv8)
2. Classifies color and car model (ResNet-18)
3. Matches vehicle to existing unauthorized tracking records:
   - Uses color + car model + temporal proximity
   - If match found:
     - Updates tracking record
     - Adds current camera to path
     - Updates timestamps
   - Shows tracking ID on blue box

### Vehicle Matching Logic

- **Camera 1**: Uses license plate for definitive identification
- **Cameras 2-4**: Uses color + car model + time window (5 minutes)
- Maintains in-memory cache for fast matching
- Updates database in real-time

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Register new user
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/me` - Get current user info

### Authorized Vehicles (Admin Only)
- `GET /api/vehicles` - List all authorized vehicles
- `POST /api/vehicles` - Add authorized vehicle
- `PUT /api/vehicles/{id}` - Update vehicle
- `DELETE /api/vehicles/{id}` - Delete vehicle

### Unauthorized Tracking
- `GET /api/tracking/unauthorized` - Get all tracking records
- `GET /api/tracking/unauthorized/active` - Get active only
- `GET /api/tracking/unauthorized/{id}` - Get specific record

### Cameras & Streaming
- `GET /api/cameras` - List available cameras
- `GET /api/stream/{camera_id}` - Stream video with detections

### Statistics
- `GET /api/stats` - Get system statistics

## Database Schema

### users
- User accounts with authentication
- Roles: admin, user

### authorized_vehicles
- Whitelisted vehicle plate numbers
- Owner information and notes

### unauthorized_vehicle_tracking
- Tracking records for unauthorized vehicles
- Camera path and timestamp history
- Active/inactive status

## Configuration

### ML Model Thresholds

Edit `backend/config.py`:

```python
DETECTION_CONFIDENCE = 0.4           # Vehicle detection threshold
PLATE_RECOGNITION_CONFIDENCE = 0.5   # LPR confidence threshold
VEHICLE_MATCH_TIME_WINDOW = 5        # Minutes for vehicle matching
```

### Video Sources

Currently uses video files. To switch to RTSP streams:

1. Edit `backend/config.py`:
```python
CAMERA_VIDEOS = {
    'camera1': 'rtsp://camera1-ip/stream',
    'camera2': 'rtsp://camera2-ip/stream',
    # ...
}
```

## Troubleshooting

### Backend Issues

**Database errors**:
```bash
# Delete and reinitialize database
rm backend/vehicle_tracking.db
python backend/init_db.py
```

**Video files not found**:
- Ensure videos are in `src/data/raw_videos/`
- Check file names match `camera1.mp4`, `camera2.mp4`, etc.

**Model loading errors**:
- Verify models are in `models/` directory
- Check CUDA availability if using GPU

### Frontend Issues

**Can't connect to backend**:
- Ensure backend is running on port 8001
- Check CORS settings in `backend/.env`

**Video streams not loading**:
- Check authentication token is valid
- Verify video files exist
- Check browser console for errors

**Build errors**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Performance Optimization

- **GPU Acceleration**: Set `gpu=True` in model initialization for faster processing
- **Video Resolution**: Lower resolution videos process faster
- **Frame Skip**: Process every Nth frame for real-time performance
- **Model Size**: Use YOLOv8n (nano) for speed, YOLOv8x for accuracy

## Security Considerations

- Change default admin password immediately
- Use strong SECRET_KEY in production
- Enable HTTPS for production deployment
- Implement rate limiting for API endpoints
- Regularly update dependencies

## Future Enhancements

- WebSocket support for real-time updates
- Face detection and recognition
- Vehicle speed estimation
- Parking duration tracking
- Alert system (email/SMS) for unauthorized vehicles
- Video recording and playback
- Advanced analytics and reports
- Mobile app integration

## License

This project is for educational and demonstration purposes.

## Credits

- **YOLOv8**: Ultralytics
- **EasyOCR**: JaidedAI
- **FastAPI**: Sebastián Ramírez
- **React**: Facebook/Meta
- **PyTorch**: Facebook AI Research

## Support

For issues and questions:
- Check troubleshooting section
- Review API documentation at http://localhost:8001/docs
- Check backend logs for errors
- Verify all dependencies are installed correctly

## Version

**Version 1.0.0** - Initial Release

---

**Built with** ❤️ **using Python, FastAPI, React, and PyTorch**




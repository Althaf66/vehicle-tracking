# Vehicle Tracking System - Backend API

FastAPI backend for the vehicle tracking system with license plate recognition.

## Features

- JWT-based authentication
- Role-based access control (admin/user)
- Authorized vehicles management (CRUD)
- Unauthorized vehicle tracking across cameras
- Real-time video streaming with vehicle detection
- License plate recognition (Camera 1 only)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Database

```bash
python backend/init_db.py
```

This will create the database tables and a default admin user:
- Username: `admin`
- Password: `admin123`

### 3. Configure Environment Variables

Edit `backend/.env` file to customize configuration:

```env
SECRET_KEY=your-secret-key-here
DATABASE_URL=vehicle_tracking.db
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### 4. Prepare Video Files

Place test video files in `src/data/raw_videos/`:
- `camera1.mp4` - Entrance camera (with license plates visible)
- `camera2.mp4` - Monitoring camera
- `camera3.mp4` - Monitoring camera
- `camera4.mp4` - Monitoring camera

### 5. Start Backend Server

```bash
python run_backend.py
```

The server will start at: http://localhost:8001

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Register new user
- `POST /api/auth/login` - Login
- `GET /api/auth/me` - Get current user info

### Authorized Vehicles (Admin only)
- `GET /api/vehicles` - List all authorized vehicles
- `POST /api/vehicles` - Add authorized vehicle
- `GET /api/vehicles/{id}` - Get vehicle details
- `PUT /api/vehicles/{id}` - Update vehicle
- `DELETE /api/vehicles/{id}` - Delete vehicle

### Unauthorized Tracking
- `GET /api/tracking/unauthorized` - Get all tracking records
- `GET /api/tracking/unauthorized/active` - Get active tracking only
- `GET /api/tracking/unauthorized/{id}` - Get specific record

### Cameras
- `GET /api/cameras` - List available cameras
- `GET /api/stream/{camera_id}` - Stream video feed

### Statistics
- `GET /api/stats` - Get system statistics

## Database Schema

### users
- id (PRIMARY KEY)
- username (UNIQUE)
- email (UNIQUE)
- password_hash
- role (admin/user)
- created_at

### authorized_vehicles
- id (PRIMARY KEY)
- plate_number (UNIQUE)
- vehicle_owner
- vehicle_type
- notes
- added_by_user_id (FOREIGN KEY)
- created_at
- updated_at

### unauthorized_vehicle_tracking
- id (PRIMARY KEY)
- tracking_id (UNIQUE)
- plate_number
- color
- car_name
- first_seen_time
- last_seen_time
- cameras_passed (JSON array)
- camera_timestamps (JSON object)
- is_active (BOOLEAN)
- notes

## Camera Processing

### Camera 1 (Entrance)
- Vehicle detection (YOLOv8)
- License plate recognition (EasyOCR)
- Authorization check against database
- Create unauthorized tracking if needed
- Box colors:
  - Green: Authorized vehicle
  - Red: Unauthorized vehicle
  - Yellow: No plate detected

### Cameras 2, 3, 4 (Monitoring)
- Vehicle detection (YOLOv8)
- Color and car name classification
- Match with unauthorized tracking records
- Update tracking with new camera sightings
- Box color: Blue (all detections)

## Development

### Project Structure

```
backend/
├── __init__.py
├── main.py              # FastAPI application
├── database.py          # Database operations
├── auth.py              # Authentication utilities
├── models.py            # Pydantic models
├── config.py            # Configuration
├── video_processor.py   # Video processing & streaming
├── init_db.py          # Database initialization
├── .env                # Environment variables
└── README.md           # This file
```

### Adding New Features

1. Define Pydantic models in `models.py`
2. Add database operations in `database.py`
3. Create API endpoints in `main.py`
4. Update video processing logic in `video_processor.py` if needed

## Troubleshooting

### Database Issues
- Delete `vehicle_tracking.db` and run `init_db.py` again

### Video Streaming Issues
- Ensure video files exist in correct location
- Check file permissions
- Verify ML models are in `models/` directory

### Authentication Issues
- Check SECRET_KEY in .env
- Verify JWT token expiration settings
- Clear browser localStorage if needed

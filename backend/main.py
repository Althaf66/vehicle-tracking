from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
from datetime import timedelta
import os
import signal
import sys
import logging

from backend.config import config
from backend.database import Database
from backend.auth import (
    authenticate_user,
    create_access_token,
    get_password_hash,
    get_current_user,
    get_current_admin_user,
    get_current_user_from_query,
    user_dict_to_response
)
from backend.models import (
    UserCreate,
    UserLogin,
    UserResponse,
    Token,
    AuthorizedVehicleCreate,
    AuthorizedVehicleUpdate,
    AuthorizedVehicleResponse,
    UnauthorizedVehicleTrackingResponse,
    UnauthorizedVehicleTrackingCreate,
    UnauthorizedVehicleTrackingUpdate,
    CameraInfo,
    CameraListResponse,
    ErrorResponse,
    MessageResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Tracking System API",
    description="Backend API for multi-camera vehicle tracking with license plate recognition",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = Database(config.DATABASE_URL)

# Graceful shutdown flag
shutdown_event = False


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting Vehicle Tracking System API...")
    logger.info(f"Database: {config.DATABASE_URL}")
    logger.info("API is ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event_handler():
    """Clean up resources on shutdown"""
    global shutdown_event
    shutdown_event = True
    logger.info("Shutting down Vehicle Tracking System API...")

    # Close database connection
    try:
        if hasattr(db, 'conn') and db.conn:
            db.conn.close()
            logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")

    logger.info("Shutdown complete")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ========== HEALTH CHECK ==========

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Vehicle Tracking API is running - CODE UPDATED!"}


@app.get("/test-auth", tags=["Health"])
async def test_auth(current_user: dict = Depends(get_current_user_from_query)):
    """Test endpoint for query-based authentication"""
    return {"status": "authenticated", "user": current_user['username']}


# ========== AUTHENTICATION ENDPOINTS ==========

@app.post("/api/auth/signup", response_model=Token, tags=["Authentication"])
async def signup(user_data: UserCreate):
    """
    Register a new user.
    Default role is 'user'. Admin users must be created manually in the database.
    """
    # Check if username already exists
    existing_user = db.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Check if email already exists
    existing_email = db.get_user_by_email(user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Hash password
    password_hash = get_password_hash(user_data.password)

    # Create user (default role: user)
    user_id = db.create_user(
        username=user_data.username,
        email=user_data.email,
        password_hash=password_hash,
        role='user'
    )

    # Get created user
    user = db.get_user_by_id(user_id)

    # Create access token
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username'], "user_id": user['id'], "role": user['role']},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_dict_to_response(user)
    )


@app.post("/api/auth/login", response_model=Token, tags=["Authentication"])
async def login(credentials: UserLogin):
    """
    Login with username/email and password.
    Returns JWT access token.
    """
    user = authenticate_user(credentials.username, credentials.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username'], "user_id": user['id'], "role": user['role']},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_dict_to_response(user)
    )


@app.get("/api/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information"""
    return user_dict_to_response(current_user)


# ========== AUTHORIZED VEHICLES ENDPOINTS (ADMIN) ==========

@app.get("/api/vehicles", response_model=List[AuthorizedVehicleResponse], tags=["Authorized Vehicles"])
async def list_authorized_vehicles(current_user: dict = Depends(get_current_user)):
    """
    Get all authorized vehicles.
    Requires authentication.
    """
    vehicles = db.get_all_authorized_vehicles()
    return vehicles


@app.post("/api/vehicles", response_model=AuthorizedVehicleResponse, tags=["Authorized Vehicles"])
async def create_authorized_vehicle(
    vehicle_data: AuthorizedVehicleCreate,
    current_user: dict = Depends(get_current_admin_user)
):
    """
    Add a new authorized vehicle.
    Requires admin role.
    """
    # Check if plate already exists
    existing = db.get_authorized_vehicle_by_plate(vehicle_data.plate_number)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vehicle with this plate number already exists"
        )

    # Create vehicle
    vehicle_id = db.add_authorized_vehicle(
        plate_number=vehicle_data.plate_number,
        vehicle_owner=vehicle_data.vehicle_owner,
        vehicle_type=vehicle_data.vehicle_type,
        notes=vehicle_data.notes,
        added_by_user_id=current_user['id']
    )

    # Get created vehicle
    vehicle = db.get_authorized_vehicle_by_plate(vehicle_data.plate_number.upper())
    return vehicle


@app.get("/api/vehicles/{vehicle_id}", response_model=AuthorizedVehicleResponse, tags=["Authorized Vehicles"])
async def get_authorized_vehicle(
    vehicle_id: int,
    current_user: dict = Depends(get_current_user)
):
    """
    Get authorized vehicle by ID.
    Requires authentication.
    """
    vehicle = None
    all_vehicles = db.get_all_authorized_vehicles()
    for v in all_vehicles:
        if v['id'] == vehicle_id:
            vehicle = v
            break

    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found"
        )

    return vehicle


@app.put("/api/vehicles/{vehicle_id}", response_model=AuthorizedVehicleResponse, tags=["Authorized Vehicles"])
async def update_authorized_vehicle(
    vehicle_id: int,
    vehicle_data: AuthorizedVehicleUpdate,
    current_user: dict = Depends(get_current_admin_user)
):
    """
    Update an authorized vehicle.
    Requires admin role.
    """
    success = db.update_authorized_vehicle(
        vehicle_id=vehicle_id,
        vehicle_owner=vehicle_data.vehicle_owner,
        vehicle_type=vehicle_data.vehicle_type,
        notes=vehicle_data.notes
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found"
        )

    # Get updated vehicle
    all_vehicles = db.get_all_authorized_vehicles()
    vehicle = None
    for v in all_vehicles:
        if v['id'] == vehicle_id:
            vehicle = v
            break

    return vehicle


@app.delete("/api/vehicles/{vehicle_id}", response_model=MessageResponse, tags=["Authorized Vehicles"])
async def delete_authorized_vehicle(
    vehicle_id: int,
    current_user: dict = Depends(get_current_admin_user)
):
    """
    Delete an authorized vehicle.
    Requires admin role.
    """
    success = db.delete_authorized_vehicle(vehicle_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found"
        )

    return MessageResponse(message="Vehicle deleted successfully")


# ========== UNAUTHORIZED VEHICLE TRACKING ENDPOINTS ==========

@app.get("/api/tracking/unauthorized", response_model=List[UnauthorizedVehicleTrackingResponse],
         tags=["Unauthorized Tracking"])
async def get_all_unauthorized_tracking(
    active_only: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """
    Get all unauthorized vehicle tracking records.
    Optionally filter to show only active vehicles.
    Only returns vehicles with confidence score >= 70%.
    Requires authentication.
    """
    records = db.get_all_unauthorized_tracking(active_only=active_only, min_confidence=70.0)
    return records


@app.get("/api/tracking/unauthorized/active", response_model=List[UnauthorizedVehicleTrackingResponse],
         tags=["Unauthorized Tracking"])
async def get_active_unauthorized_tracking(current_user: dict = Depends(get_current_user)):
    """
    Get only active unauthorized vehicle tracking records.
    Only returns vehicles with confidence score >= 70%.
    Requires authentication.
    """
    records = db.get_all_unauthorized_tracking(active_only=True, min_confidence=70.0)
    return records


@app.get("/api/tracking/unauthorized/{tracking_id}", response_model=UnauthorizedVehicleTrackingResponse,
         tags=["Unauthorized Tracking"])
async def get_unauthorized_tracking_by_id(
    tracking_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get specific unauthorized vehicle tracking record by tracking ID.
    Requires authentication.
    """
    record = db.get_unauthorized_tracking_by_id(tracking_id)

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tracking record not found"
        )

    return record


@app.post("/api/tracking/unauthorized", response_model=UnauthorizedVehicleTrackingResponse,
          tags=["Unauthorized Tracking"])
async def create_unauthorized_tracking(tracking_data: UnauthorizedVehicleTrackingCreate):
    """
    Internal endpoint to create a new unauthorized vehicle tracking record.
    Called by video processor when unauthorized vehicle is detected.
    No authentication required (internal use).
    """
    # Check if tracking ID already exists
    existing = db.get_unauthorized_tracking_by_id(tracking_data.tracking_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tracking record with this ID already exists"
        )

    # Create tracking record
    record_id = db.add_unauthorized_tracking(
        tracking_id=tracking_data.tracking_id,
        plate_number=tracking_data.plate_number,
        color=tracking_data.color,
        car_name=tracking_data.car_name,
        confidence_score=tracking_data.confidence_score,
        camera_id=tracking_data.camera_id
    )

    # Get created record
    record = db.get_unauthorized_tracking_by_id(tracking_data.tracking_id)
    return record


@app.put("/api/tracking/unauthorized", response_model=UnauthorizedVehicleTrackingResponse,
         tags=["Unauthorized Tracking"])
async def update_unauthorized_tracking(tracking_data: UnauthorizedVehicleTrackingUpdate):
    """
    Internal endpoint to update an unauthorized vehicle tracking record.
    Called by video processor when vehicle is seen in another camera.
    No authentication required (internal use).
    """
    success = db.update_unauthorized_tracking(
        tracking_id=tracking_data.tracking_id,
        camera_id=tracking_data.camera_id
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tracking record not found"
        )

    # Get updated record
    record = db.get_unauthorized_tracking_by_id(tracking_data.tracking_id)
    return record


# ========== CAMERA ENDPOINTS ==========

@app.get("/api/cameras", response_model=CameraListResponse, tags=["Cameras"])
async def list_cameras(current_user: dict = Depends(get_current_user)):
    """
    Get list of available cameras with their status.
    Requires authentication.
    """
    cameras = []

    for camera_id, video_path in config.CAMERA_VIDEOS.items():
        camera_type = 'entrance' if camera_id == 'camera1' else 'monitoring'
        has_lpr = camera_id == 'camera1'
        camera_name = f"Camera {camera_id[-1]}"

        if camera_type == 'entrance':
           camera_name += " - Entrance (LPR)"
        else:
           camera_name += " - Monitoring"

    # Check if video file exists OR if it's a live stream URL
        if video_path.startswith("rtsp"):
           video_status = 'online'
        else:
           video_status = 'online' if os.path.exists(video_path) else 'offline'

        cameras.append(CameraInfo(
           camera_id=camera_id,
           camera_name=camera_name,
           camera_type=camera_type,
           has_lpr=has_lpr,
           status=video_status
       ))

    return CameraListResponse(cameras=cameras)


@app.get("/api/stream/{camera_id}", tags=["Video Streaming"])
async def stream_camera(camera_id: str, token: str):
    """
    Stream processed video from specified camera.
    Returns MJPEG stream with vehicle detection and tracking.
    Requires authentication (token passed as query parameter).
    """
    # Validate token
    from backend.auth import decode_access_token
    token_data = decode_access_token(token)
    if token_data is None or token_data.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify user exists
    user = db.get_user_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if camera_id not in config.CAMERA_VIDEOS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )

    video_path = config.CAMERA_VIDEOS[camera_id]
    if not video_path.startswith("rtsp") and not os.path.exists(video_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Camera {camera_id} is offline - video file not found"
        )

    # Import video processor (lazy import to avoid circular dependencies)
    from backend.video_processor import stream_video

    camera_type = 'entrance' if camera_id == 'camera1' else 'monitoring'

    return StreamingResponse(
        stream_video(video_path, camera_id, camera_type),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


# ========== STATISTICS ENDPOINT ==========

@app.get("/api/stats", tags=["Statistics"])
async def get_statistics(current_user: dict = Depends(get_current_user)):
    """
    Get system statistics.
    Only counts vehicles with confidence score >= 70%.
    Requires authentication.
    """
    # Get counts (filtered by min_confidence >= 70%)
    all_authorized = db.get_all_authorized_vehicles()
    all_unauthorized = db.get_all_unauthorized_tracking(min_confidence=70.0)
    active_unauthorized = db.get_all_unauthorized_tracking(active_only=True, min_confidence=70.0)

    return {
        "authorized_vehicles_count": len(all_authorized),
        "total_unauthorized_tracked": len(all_unauthorized),
        "active_unauthorized_count": len(active_unauthorized),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict
from datetime import datetime


# ========== USER MODELS ==========

class UserCreate(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class UserResponse(BaseModel):
    """User information response"""
    id: int
    username: str
    email: str
    role: str
    created_at: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenData(BaseModel):
    """Data extracted from JWT token"""
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None


# ========== AUTHORIZED VEHICLE MODELS ==========

class AuthorizedVehicleCreate(BaseModel):
    """Create authorized vehicle request"""
    plate_number: str = Field(..., min_length=1, max_length=15)
    vehicle_owner: Optional[str] = None
    vehicle_type: Optional[str] = Field(None, pattern="^(car|truck|bus|other)$")
    notes: Optional[str] = None


class AuthorizedVehicleUpdate(BaseModel):
    """Update authorized vehicle request"""
    vehicle_owner: Optional[str] = None
    vehicle_type: Optional[str] = Field(None, pattern="^(car|truck|bus|other)$")
    notes: Optional[str] = None


class AuthorizedVehicleResponse(BaseModel):
    """Authorized vehicle response"""
    id: int
    plate_number: str
    vehicle_owner: Optional[str]
    vehicle_type: Optional[str]
    notes: Optional[str]
    added_by_user_id: Optional[int]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


# ========== UNAUTHORIZED VEHICLE TRACKING MODELS ==========

class UnauthorizedVehicleTrackingResponse(BaseModel):
    """Unauthorized vehicle tracking response"""
    id: int
    tracking_id: str
    plate_number: Optional[str]
    color: Optional[str]
    car_name: Optional[str]
    confidence_score: Optional[float]
    first_seen_time: str
    last_seen_time: str
    cameras_passed: List[str]
    camera_timestamps: Dict[str, str]
    is_active: bool
    notes: Optional[str]

    class Config:
        from_attributes = True


class UnauthorizedVehicleTrackingCreate(BaseModel):
    """Create unauthorized vehicle tracking request"""
    tracking_id: str
    plate_number: Optional[str] = None
    color: Optional[str] = None
    car_name: Optional[str] = None
    confidence_score: Optional[float] = None
    camera_id: str = 'camera1'


class UnauthorizedVehicleTrackingUpdate(BaseModel):
    """Update unauthorized vehicle tracking request"""
    tracking_id: str
    camera_id: str


# ========== CAMERA MODELS ==========

class CameraInfo(BaseModel):
    """Camera information"""
    camera_id: str
    camera_name: str
    camera_type: str  # 'entrance' or 'monitoring'
    has_lpr: bool  # License Plate Recognition capability
    status: str  # 'online' or 'offline'


class CameraListResponse(BaseModel):
    """List of available cameras"""
    cameras: List[CameraInfo]


# ========== DETECTION MODELS ==========

class VehicleDetection(BaseModel):
    """Real-time vehicle detection information"""
    vehicle_id: Optional[int]
    tracking_id: Optional[str]
    bbox: List[int]  # [x1, y1, x2, y2]
    color: str
    car_name: str
    confidence: float
    license_plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    is_authorized: Optional[bool] = None
    status: str  # 'TRACKED', 'NEW', 'LOW_CONFIDENCE'
    camera_id: str
    timestamp: str


# ========== ERROR MODELS ==========

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str

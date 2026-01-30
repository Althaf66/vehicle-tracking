from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Query, Request
from fastapi.security import OAuth2PasswordBearer
from backend.config import config
from backend.database import Database
from backend.models import TokenData, UserResponse

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Database instance
db = Database(config.DATABASE_URL)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token

    Args:
        data: Dictionary containing user data to encode
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """
    Decode and verify JWT access token

    Args:
        token: JWT token string

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        user_id: int = payload.get("user_id")
        username: str = payload.get("sub")
        role: str = payload.get("role")

        if username is None or user_id is None:
            return None

        return TokenData(user_id=user_id, username=username, role=role)
    except JWTError:
        return None


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Authenticate a user with username and password

    Args:
        username: Username or email
        password: Plain text password

    Returns:
        User dict if authenticated, None otherwise
    """
    # Try to find user by username
    user = db.get_user_by_username(username)

    # If not found, try by email
    if not user:
        user = db.get_user_by_email(username)

    if not user:
        return None

    if not verify_password(password, user['password_hash']):
        return None

    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Get current authenticated user from JWT token

    Args:
        token: JWT token from Authorization header

    Returns:
        User dict

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = decode_access_token(token)
    if token_data is None or token_data.user_id is None:
        raise credentials_exception

    user = db.get_user_by_id(token_data.user_id)
    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Get current active user (for future use if we add user deactivation)

    Args:
        current_user: User dict from get_current_user

    Returns:
        User dict
    """
    return current_user


async def get_current_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Get current user and verify they have admin role

    Args:
        current_user: User dict from get_current_user

    Returns:
        User dict if user is admin

    Raises:
        HTTPException: If user is not an admin
    """
    if current_user['role'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin role required."
        )
    return current_user


async def get_current_user_from_query(request: Request) -> dict:
    """
    Get current authenticated user from JWT token passed as query parameter.
    This is used for video streaming where Authorization headers cannot be set (e.g., <img> tags).

    Args:
        request: FastAPI Request object

    Returns:
        User dict

    Raises:
        HTTPException: If token is invalid or user not found
    """
    print(f"[DEBUG] get_current_user_from_query called")
    print(f"[DEBUG] Query params: {request.query_params}")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Get token from query parameters
    token = request.query_params.get("token")
    print(f"[DEBUG] Token from query: {token[:20] if token else 'None'}...")

    if token is None:
        print("[DEBUG] Token is None, raising exception")
        raise credentials_exception

    token_data = decode_access_token(token)
    print(f"[DEBUG] Token decoded: {token_data}")

    if token_data is None or token_data.user_id is None:
        print("[DEBUG] Token data invalid, raising exception")
        raise credentials_exception

    user = db.get_user_by_id(token_data.user_id)
    print(f"[DEBUG] User found: {user.get('username') if user else 'None'}")

    if user is None:
        print("[DEBUG] User not found, raising exception")
        raise credentials_exception

    print(f"[DEBUG] Returning user successfully")
    return user


def user_dict_to_response(user_dict: dict) -> UserResponse:
    """
    Convert user dictionary to UserResponse model

    Args:
        user_dict: User data dictionary from database

    Returns:
        UserResponse model
    """
    return UserResponse(
        id=user_dict['id'],
        username=user_dict['username'],
        email=user_dict['email'],
        role=user_dict['role'],
        created_at=user_dict['created_at']
    )

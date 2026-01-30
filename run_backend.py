"""
Backend server startup script.
Runs the FastAPI application with uvicorn.
"""

import uvicorn
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """Start the FastAPI backend server"""
    print("=" * 60)
    print("Vehicle Tracking System - Backend Server")
    print("=" * 60)
    print("\nStarting server...")
    print("API will be available at: http://localhost:8001")
    print("API documentation at: http://localhost:8001/docs")
    print("Interactive API explorer at: http://localhost:8001/redoc")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    print()

    # Run uvicorn server
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )


if __name__ == "__main__":
    main()

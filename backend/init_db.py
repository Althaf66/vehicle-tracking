"""
Database initialization script.
Creates tables and adds default admin user.
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import Database
from auth import get_password_hash


def initialize_database():
    """Initialize database with tables and default admin user"""
    print("=" * 60)
    print("Vehicle Tracking System - Database Initialization")
    print("=" * 60)

    # Initialize database (creates tables)
    db = Database('vehicle_tracking.db')
    print("\nDatabase tables created successfully!")

    # Check if admin user already exists
    existing_admin = db.get_user_by_username('admin')

    if existing_admin:
        print("\nAdmin user already exists.")
        print(f"  Username: admin")
        print(f"  Email: {existing_admin['email']}")
    else:
        # Create default admin user
        print("\nCreating default admin user...")
        admin_password = 'admin123'
        password_hash = get_password_hash(admin_password)

        admin_id = db.create_user(
            username='admin',
            email='admin@vehicletracking.com',
            password_hash=password_hash,
            role='admin'
        )

        print(f"Admin user created successfully!")
        print(f"  Username: admin")
        print(f"  Password: {admin_password}")
        print(f"  Email: admin@vehicletracking.com")
        print(f"\n⚠️  IMPORTANT: Please change the admin password after first login!")

    # Display summary
    print("\n" + "=" * 60)
    print("Database Initialization Complete!")
    print("=" * 60)
    print("\nDatabase Tables:")
    print("  1. users - User accounts with authentication")
    print("  2. authorized_vehicles - Authorized vehicle plate numbers")
    print("  3. unauthorized_vehicle_tracking - Tracking of unauthorized vehicles")

    print("\nDefault Credentials:")
    print("  Admin Username: admin")
    print("  Admin Password: admin123")

    print("\nNext Steps:")
    print("  1. Run the backend server: python run_backend.py")
    print("  2. Access the API at: http://localhost:8001")
    print("  3. View API documentation at: http://localhost:8001/docs")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        initialize_database()
    except Exception as e:
        print(f"\nError during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

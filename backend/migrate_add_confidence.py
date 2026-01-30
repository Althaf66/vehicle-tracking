"""
Database migration script to add confidence_score column to unauthorized_vehicle_tracking table.
Run this script if you have an existing database without the confidence_score column.
"""

import sqlite3
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import config


def migrate_database():
    """Add confidence_score column to existing database"""
    print("=" * 60)
    print("Database Migration: Adding confidence_score column")
    print("=" * 60)

    db_path = config.DATABASE_URL

    if not os.path.exists(db_path):
        print(f"\nError: Database not found at {db_path}")
        print("Please run init_db.py first to create the database.")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(unauthorized_vehicle_tracking)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'confidence_score' in columns:
            print("\n✓ confidence_score column already exists. No migration needed.")
            conn.close()
            return True

        # Add the column
        print("\nAdding confidence_score column...")
        cursor.execute('''
            ALTER TABLE unauthorized_vehicle_tracking
            ADD COLUMN confidence_score REAL
        ''')

        conn.commit()
        print("✓ Successfully added confidence_score column")

        # Verify the column was added
        cursor.execute("PRAGMA table_info(unauthorized_vehicle_tracking)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'confidence_score' in columns:
            print("✓ Migration verified successfully")
            print("\nNote: Existing records will have NULL confidence scores.")
            print("New detections will automatically include confidence scores.")
        else:
            print("✗ Migration verification failed")
            return False

        conn.close()

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("Migration completed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)

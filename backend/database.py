import sqlite3
from datetime import datetime
import json
from contextlib import contextmanager
from typing import Optional, List, Dict, Any


class Database:
    """
    SQLite database handler for vehicle tracking system.
    Manages three tables: users, authorized_vehicles, and unauthorized_vehicle_tracking.
    """

    def __init__(self, db_path='vehicle_tracking.db'):
        self.db_path = db_path
        self._create_tables()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _create_tables(self):
        """Create all required tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK (role IN ('admin', 'user'))
                )
            ''')

            # Create authorized_vehicles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS authorized_vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE NOT NULL,
                    vehicle_owner TEXT,
                    vehicle_type TEXT,
                    notes TEXT,
                    added_by_user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (added_by_user_id) REFERENCES users (id),
                    CHECK (vehicle_type IN ('car', 'truck', 'bus', 'other') OR vehicle_type IS NULL)
                )
            ''')

            # Create unauthorized_vehicle_tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unauthorized_vehicle_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracking_id TEXT UNIQUE NOT NULL,
                    plate_number TEXT,
                    color TEXT,
                    car_name TEXT,
                    confidence_score REAL,
                    first_seen_time TIMESTAMP NOT NULL,
                    last_seen_time TIMESTAMP NOT NULL,
                    cameras_passed TEXT NOT NULL,
                    camera_timestamps TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    notes TEXT
                )
            ''')

            # Add confidence_score column if it doesn't exist (for existing databases)
            try:
                cursor.execute('''
                    ALTER TABLE unauthorized_vehicle_tracking
                    ADD COLUMN confidence_score REAL
                ''')
                print("Added confidence_score column to existing table")
            except sqlite3.OperationalError:
                # Column already exists
                pass

            # Create indexes for better query performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_plate_number
                ON authorized_vehicles(plate_number)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_tracking_id
                ON unauthorized_vehicle_tracking(tracking_id)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_is_active
                ON unauthorized_vehicle_tracking(is_active)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_username
                ON users(username)
            ''')

            conn.commit()
            print("Database tables created successfully")

    # ========== USER OPERATIONS ==========

    def create_user(self, username: str, email: str, password_hash: str, role: str = 'user') -> int:
        """Create a new user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, role))
            return cursor.lastrowid

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== AUTHORIZED VEHICLES OPERATIONS ==========

    def add_authorized_vehicle(self, plate_number: str, vehicle_owner: str = None,
                              vehicle_type: str = None, notes: str = None,
                              added_by_user_id: int = None) -> int:
        """Add a new authorized vehicle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO authorized_vehicles
                (plate_number, vehicle_owner, vehicle_type, notes, added_by_user_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (plate_number.upper(), vehicle_owner, vehicle_type, notes, added_by_user_id))
            return cursor.lastrowid

    def get_authorized_vehicle_by_plate(self, plate_number: str) -> Optional[Dict[str, Any]]:
        """Check if a vehicle is authorized by plate number"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM authorized_vehicles WHERE plate_number = ?
            ''', (plate_number.upper(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_authorized_vehicles(self) -> List[Dict[str, Any]]:
        """Get all authorized vehicles"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM authorized_vehicles ORDER BY created_at DESC')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_authorized_vehicle(self, vehicle_id: int, vehicle_owner: str = None,
                                 vehicle_type: str = None, notes: str = None) -> bool:
        """Update an authorized vehicle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE authorized_vehicles
                SET vehicle_owner = ?, vehicle_type = ?, notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (vehicle_owner, vehicle_type, notes, vehicle_id))
            return cursor.rowcount > 0

    def delete_authorized_vehicle(self, vehicle_id: int) -> bool:
        """Delete an authorized vehicle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM authorized_vehicles WHERE id = ?', (vehicle_id,))
            return cursor.rowcount > 0

    def is_vehicle_authorized(self, plate_number: str) -> bool:
        """Check if a vehicle is authorized"""
        return self.get_authorized_vehicle_by_plate(plate_number) is not None

    # ========== UNAUTHORIZED VEHICLE TRACKING OPERATIONS ==========

    def add_unauthorized_tracking(self, tracking_id: str, plate_number: str = None,
                                 color: str = None, car_name: str = None,
                                 confidence_score: float = None,
                                 first_seen_time: datetime = None,
                                 camera_id: str = 'camera1') -> int:
        """Add a new unauthorized vehicle tracking record"""
        if first_seen_time is None:
            first_seen_time = datetime.now()

        cameras_passed = json.dumps([camera_id])
        camera_timestamps = json.dumps({camera_id: first_seen_time.isoformat()})

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO unauthorized_vehicle_tracking
                (tracking_id, plate_number, color, car_name, confidence_score,
                 first_seen_time, last_seen_time, cameras_passed, camera_timestamps, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            ''', (tracking_id, plate_number, color, car_name, confidence_score,
                  first_seen_time, first_seen_time, cameras_passed, camera_timestamps))
            return cursor.lastrowid

    def update_unauthorized_tracking(self, tracking_id: str, camera_id: str,
                                    timestamp: datetime = None) -> bool:
        """Update an unauthorized vehicle tracking record with new camera sighting"""
        if timestamp is None:
            timestamp = datetime.now()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get current record
            cursor.execute('''
                SELECT cameras_passed, camera_timestamps
                FROM unauthorized_vehicle_tracking
                WHERE tracking_id = ?
            ''', (tracking_id,))
            row = cursor.fetchone()

            if not row:
                return False

            # Parse existing data
            cameras_passed = json.loads(row['cameras_passed'])
            camera_timestamps = json.loads(row['camera_timestamps'])

            # Add new camera if not already present
            if camera_id not in cameras_passed:
                cameras_passed.append(camera_id)

            # Update timestamp for this camera
            camera_timestamps[camera_id] = timestamp.isoformat()

            # Update database
            cursor.execute('''
                UPDATE unauthorized_vehicle_tracking
                SET cameras_passed = ?, camera_timestamps = ?,
                    last_seen_time = ?, is_active = TRUE
                WHERE tracking_id = ?
            ''', (json.dumps(cameras_passed), json.dumps(camera_timestamps),
                  timestamp, tracking_id))

            return cursor.rowcount > 0

    def get_unauthorized_tracking_by_id(self, tracking_id: str) -> Optional[Dict[str, Any]]:
        """Get unauthorized vehicle tracking record by tracking ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM unauthorized_vehicle_tracking WHERE tracking_id = ?
            ''', (tracking_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['cameras_passed'] = json.loads(result['cameras_passed'])
                result['camera_timestamps'] = json.loads(result['camera_timestamps'])
                return result
            return None

    def find_unauthorized_tracking_by_features(self, color: str, car_name: str,
                                              time_window_minutes: int = 5) -> Optional[Dict[str, Any]]:
        """
        Find an active unauthorized vehicle tracking record by color and car name
        within a time window (for matching across cameras).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM unauthorized_vehicle_tracking
                WHERE color = ? AND car_name = ? AND is_active = TRUE
                AND datetime(last_seen_time) >= datetime('now', '-' || ? || ' minutes')
                ORDER BY last_seen_time DESC
                LIMIT 1
            ''', (color, car_name, time_window_minutes))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['cameras_passed'] = json.loads(result['cameras_passed'])
                result['camera_timestamps'] = json.loads(result['camera_timestamps'])
                return result
            return None

    def get_all_unauthorized_tracking(self, active_only: bool = False,
                                     min_confidence: float = None) -> List[Dict[str, Any]]:
        """Get all unauthorized vehicle tracking records"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = 'SELECT * FROM unauthorized_vehicle_tracking WHERE 1=1'
            params = []

            if active_only:
                query += ' AND is_active = TRUE'

            if min_confidence is not None:
                query += ' AND confidence_score >= ?'
                params.append(min_confidence)

            query += ' ORDER BY last_seen_time DESC'

            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result['cameras_passed'] = json.loads(result['cameras_passed'])
                result['camera_timestamps'] = json.loads(result['camera_timestamps'])
                results.append(result)
            return results

    def deactivate_unauthorized_tracking(self, tracking_id: str) -> bool:
        """Mark an unauthorized vehicle as inactive (left the premises)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE unauthorized_vehicle_tracking
                SET is_active = FALSE
                WHERE tracking_id = ?
            ''', (tracking_id,))
            return cursor.rowcount > 0

    def cleanup_old_inactive_records(self, days: int = 30) -> int:
        """Delete inactive tracking records older than specified days"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM unauthorized_vehicle_tracking
                WHERE is_active = FALSE
                AND datetime(last_seen_time) < datetime('now', '-' || ? || ' days')
            ''', (days,))
            return cursor.rowcount


def initialize_database(db_path='vehicle_tracking.db'):
    """Initialize the database with tables"""
    db = Database(db_path)
    print(f"Database initialized at: {db_path}")
    return db


if __name__ == "__main__":
    # Test database creation
    db = initialize_database('test_vehicle_tracking.db')
    print("\nDatabase test successful!")
    print("Tables created:")
    print("  1. users")
    print("  2. authorized_vehicles")
    print("  3. unauthorized_vehicle_tracking")

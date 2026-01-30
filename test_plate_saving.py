"""
Diagnostic script to test license plate recognition and database saving
"""
import sys
import os

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.dirname(__file__))

from backend.database import Database
from src.license_plate_recognizer import LicensePlateRecognizer

def test_database_operations():
    """Test database operations for saving plates"""
    print("=" * 70)
    print("DATABASE OPERATIONS TEST")
    print("=" * 70)

    # Initialize database
    db = Database('vehicle_tracking.db')

    # Test adding an unauthorized vehicle with a plate
    print("\n1. Testing add_unauthorized_tracking with plate...")
    tracking_id = "TEST-SILVER-HONDA-CR-V"
    plate_number = "KL01AA3456"

    try:
        # Check if record exists
        existing = db.get_unauthorized_tracking_by_id(tracking_id)
        if existing:
            print(f"   Record already exists: {tracking_id}")
        else:
            # Add new record
            record_id = db.add_unauthorized_tracking(
                tracking_id=tracking_id,
                plate_number=plate_number,
                color="SILVER",
                car_name="HONDA-CR-V",
                confidence_score=85.5,
                camera_id="camera1"
            )
            print(f"   [OK] Record added successfully with ID: {record_id}")
    except Exception as e:
        print(f"   [ERR] Error adding record: {e}")

    # Verify the record was saved
    print("\n2. Verifying record was saved...")
    try:
        record = db.get_unauthorized_tracking_by_id(tracking_id)
        if record:
            print(f"   [OK] Record found:")
            print(f"      - Tracking ID: {record['tracking_id']}")
            print(f"      - Plate: {record['plate_number']}")
            print(f"      - Color: {record['color']}")
            print(f"      - Car: {record['car_name']}")
            print(f"      - Confidence: {record['confidence_score']}")
        else:
            print(f"   [ERR] Record not found!")
    except Exception as e:
        print(f"   [ERR] Error retrieving record: {e}")

    # List all records
    print("\n3. Listing all unauthorized vehicle records...")
    try:
        records = db.get_all_unauthorized_tracking()
        print(f"   Total records: {len(records)}")
        for i, r in enumerate(records[:5], 1):
            print(f"   {i}. ID: {r['tracking_id'][:30]}, Plate: {r.get('plate_number', 'None')}, "
                  f"Color: {r.get('color')}, Car: {r.get('car_name')}")
    except Exception as e:
        print(f"   [ERR] Error listing records: {e}")


def test_plate_validation():
    """Test plate validation with different formats"""
    print("\n" + "=" * 70)
    print("PLATE VALIDATION TEST")
    print("=" * 70)

    # Test with strict validation OFF
    print("\n1. Testing with STRICT validation OFF...")
    recognizer = LicensePlateRecognizer(gpu_enabled=False, strict_validation=False)

    test_plates = [
        "KL01AA3456",  # Valid Indian format
        "ABC123",      # Non-standard format
        "DL01AB1234",  # Valid Indian format
        "12345",       # Numbers only
        "ABCDEF",      # Letters only
    ]

    for plate in test_plates:
        validated, is_valid = recognizer.validate_indian_plate_format(plate)
        print(f"   Input: {plate:15} -> Output: {validated:15} Valid: {is_valid}")

    # Test with strict validation ON
    print("\n2. Testing with STRICT validation ON...")
    recognizer_strict = LicensePlateRecognizer(gpu_enabled=False, strict_validation=True)

    for plate in test_plates:
        validated, is_valid = recognizer_strict.validate_indian_plate_format(plate)
        print(f"   Input: {plate:15} -> Output: {validated:15} Valid: {is_valid}")


def test_config():
    """Test configuration loading"""
    print("\n" + "=" * 70)
    print("CONFIGURATION TEST")
    print("=" * 70)

    try:
        from backend.config import config

        print(f"\nLPR Configuration:")
        print(f"   PLATE_RECOGNITION_CONFIDENCE: {config.PLATE_RECOGNITION_CONFIDENCE}")
        print(f"   INDIAN_PLATE_FORMAT_VALIDATION: {config.INDIAN_PLATE_FORMAT_VALIDATION}")
        print(f"   STRICT_PLATE_VALIDATION: {config.STRICT_PLATE_VALIDATION}")
        print(f"   PLATE_AGGREGATION_WINDOW: {config.PLATE_AGGREGATION_WINDOW}")
        print(f"   CHARACTER_CONFUSION_CORRECTION: {config.CHARACTER_CONFUSION_CORRECTION}")
    except Exception as e:
        print(f"   [ERR] Error loading config: {e}")


def main():
    """Run all diagnostic tests"""
    print("\n" + "=" * 70)
    print("LICENSE PLATE RECOGNITION & DATABASE DIAGNOSTIC")
    print("=" * 70)

    test_config()
    test_database_operations()
    test_plate_validation()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("TROUBLESHOOTING GUIDE")
    print("=" * 70)
    print("""
If plates are not being saved:

1. Check backend console logs for:
   - [LPR] Plate found but confidence too low
   - [LPR] Invalid format rejected
   - [LPR] Warning: Non-standard format accepted

2. Verify configuration in backend/.env:
   - STRICT_PLATE_VALIDATION=false (to accept all plates)
   - PLATE_RECOGNITION_CONFIDENCE=0.35 (lower for more detections)

3. If no plates are detected at all:
   - Check if EasyOCR is working
   - Verify video quality is good
   - Check if plates are visible in the video

4. To debug further:
   - Start backend with: python backend/run_backend.py
   - Watch console for [LPR] messages
   - Check if vehicles are being detected at all
    """)


if __name__ == "__main__":
    main()

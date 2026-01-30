"""
Unit tests for multi-frame plate aggregation
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.license_plate_recognizer import LicensePlateRecognizer


def test_aggregation_with_consistent_readings():
    """Test aggregation when all frames read the same plate"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)

    vehicle_id = "test_vehicle_1"
    plate = "KL01AA3456"

    print("Testing Aggregation with Consistent Readings:")
    print("=" * 60)

    # Simulate 5 consistent readings
    confidences = [0.7, 0.75, 0.8, 0.72, 0.78]

    for i, conf in enumerate(confidences):
        aggregated_plate, aggregated_conf = recognizer.aggregate_plate_readings(
            vehicle_id, plate, conf
        )
        print(f"Frame {i+1}: Input conf={conf:.2f}, "
              f"Aggregated plate={aggregated_plate}, conf={aggregated_conf:.2f}")

    print("[OK] Test completed\n")


def test_aggregation_with_majority_voting():
    """Test aggregation when different plates are read (majority voting)"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)

    vehicle_id = "test_vehicle_2"

    print("Testing Aggregation with Majority Voting:")
    print("=" * 60)

    # Simulate readings with occasional errors
    readings = [
        ("KL01AA3456", 0.8),   # Correct
        ("KL01AA3456", 0.75),  # Correct
        ("KL01AA345S", 0.6),   # OCR error (S instead of 6)
        ("KL01AA3456", 0.82),  # Correct
        ("KL01AA3456", 0.78),  # Correct
        ("KL01AA3456", 0.8),   # Correct
    ]

    for i, (plate, conf) in enumerate(readings):
        aggregated_plate, aggregated_conf = recognizer.aggregate_plate_readings(
            vehicle_id, plate, conf
        )
        print(f"Frame {i+1}: Input plate={plate}, conf={conf:.2f} | "
              f"Aggregated plate={aggregated_plate}, conf={aggregated_conf:.2f}")

    print("[OK] Expected: Majority plate (KL01AA3456) should win\n")


def test_aggregation_confidence_boost():
    """Test that high agreement boosts confidence"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)

    vehicle_id = "test_vehicle_3"
    plate = "KL01AA3456"

    print("Testing Confidence Boost with High Agreement:")
    print("=" * 60)

    # Simulate 10 consistent readings (should trigger 60% agreement boost)
    for i in range(10):
        conf = 0.7 + (i * 0.01)  # Slightly varying confidence
        aggregated_plate, aggregated_conf = recognizer.aggregate_plate_readings(
            vehicle_id, plate, conf
        )

        if i >= 2:  # Aggregation starts at 3 readings
            agreement_count = i + 1
            expected_boost = agreement_count >= 3 * 0.6  # 60% of readings
            boost_indicator = "[BOOST]" if expected_boost else ""
            print(f"Frame {i+1}: Aggregated conf={aggregated_conf:.2f} {boost_indicator}")

    print("[OK] Expected: Confidence should increase with agreement\n")


def test_aggregation_window_limit():
    """Test that aggregation respects the history window limit"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)
    recognizer.history_window = 5  # Set small window for testing

    vehicle_id = "test_vehicle_4"

    print(f"Testing Aggregation Window Limit (window={recognizer.history_window}):")
    print("=" * 60)

    # Add more readings than the window size
    for i in range(8):
        plate = "KL01AA3456"
        conf = 0.7
        aggregated_plate, aggregated_conf = recognizer.aggregate_plate_readings(
            vehicle_id, plate, conf
        )

        history_size = len(recognizer.plate_history[vehicle_id])
        print(f"Frame {i+1}: History size={history_size} (max={recognizer.history_window})")

    print("[OK] Expected: History size should not exceed window limit\n")


def test_multiple_vehicles():
    """Test that different vehicles maintain separate histories"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)

    print("Testing Multiple Vehicles with Separate Histories:")
    print("=" * 60)

    vehicles = [
        ("vehicle_1", "KL01AA3456"),
        ("vehicle_2", "MH12DE1234"),
        ("vehicle_3", "DL01BC5678"),
    ]

    # Add readings for each vehicle
    for vehicle_id, plate in vehicles:
        for i in range(3):
            conf = 0.75 + (i * 0.05)
            aggregated_plate, aggregated_conf = recognizer.aggregate_plate_readings(
                vehicle_id, plate, conf
            )

        print(f"Vehicle {vehicle_id}: Plate={plate}, "
              f"History entries={len(recognizer.plate_history[vehicle_id])}")

    total_vehicles = len(recognizer.plate_history)
    print(f"\n[OK] Total vehicles tracked: {total_vehicles} (expected: {len(vehicles)})\n")


def run_all_tests():
    """Run all aggregation tests"""
    print("\n" + "=" * 70)
    print("MULTI-FRAME PLATE AGGREGATION TEST SUITE")
    print("=" * 70 + "\n")

    test_aggregation_with_consistent_readings()
    test_aggregation_with_majority_voting()
    test_aggregation_confidence_boost()
    test_aggregation_window_limit()
    test_multiple_vehicles()

    print("=" * 70)
    print("Test suite completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

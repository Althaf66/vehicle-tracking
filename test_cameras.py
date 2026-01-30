"""
Test script to verify all 4 cameras can be processed.
Run this to ensure all camera video files are accessible and processable.
"""

import sys
import os
import cv2

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.config import config


def test_camera_video(camera_id, video_path):
    """Test if a camera video file can be opened and read"""
    print(f"\nTesting {camera_id}:")
    print(f"  Path: {video_path}")

    # Check if file exists
    if not os.path.exists(video_path):
        print(f"  [X] ERROR: Video file not found!")
        return False

    print(f"  OK File exists")

    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [X] ERROR: Could not open video file!")
        return False

    print(f"  OK Video file can be opened")

    # Try to read first frame
    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"  [X] ERROR: Could not read frame from video!")
        cap.release()
        return False

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  OK Video is readable")
    print(f"  Info: Resolution: {width}x{height}")
    print(f"  Info: FPS: {fps:.2f}")
    print(f"  Info: Total frames: {frame_count}")

    cap.release()
    return True


def main():
    print("=" * 60)
    print("Camera Video Files Test")
    print("=" * 60)

    results = {}

    for camera_id, video_path in config.CAMERA_VIDEOS.items():
        results[camera_id] = test_camera_video(camera_id, video_path)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for camera_id, passed in results.items():
        status = "[OK] PASS" if passed else "[X] FAIL"
        camera_type = "Entrance (LPR)" if camera_id == "camera1" else "Monitoring"
        print(f"{status} - {camera_id} ({camera_type})")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("[OK] All cameras are working correctly!")
        print("\nYou can now start the backend:")
        print("  python run_backend.py")
    else:
        print("[X] Some cameras failed the test.")
        print("\nPlease check the video files and paths.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

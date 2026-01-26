import cv2
import torch
from ultralytics import YOLO
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_gpu_available():
    """Check if GPU is available and raise error if not"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ERROR: GPU not available! This script requires GPU to run.\n"
            "Please ensure CUDA is properly installed and a compatible GPU is available."
        )

    device = torch.cuda.get_device_name(0)
    print(f"GPU detected: {device}")
    print(f"CUDA version: {torch.version.cuda}")
    return True

def detect_vehicle_segments(video_path, confidence_threshold=0.5, min_segment_duration=0.5):
    """
    Detect continuous segments in video where vehicles are present

    Args:
        video_path: Path to input video
        confidence_threshold: Minimum confidence for vehicle detection (0-1)
        min_segment_duration: Minimum duration in seconds for a segment to be kept

    Returns:
        List of tuples (start_frame, end_frame) representing vehicle segments
    """
    # Check GPU availability
    check_gpu_available()

    # Load YOLO model on GPU
    model = YOLO('yolov8n.pt')
    model.to('cuda')

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_frames = int(fps * min_segment_duration)

    print(f"\nProcessing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Minimum segment duration: {min_segment_duration}s ({min_frames} frames)")
    print(f"Confidence threshold: {confidence_threshold}")

    # Track frames with vehicles
    frames_with_vehicles = []
    frame_idx = 0

    # Process video frame by frame
    with tqdm(total=total_frames, desc="Detecting vehicles") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect vehicles (classes: 2=car, 5=bus, 7=truck)
            results = model(frame, classes=[2, 5, 7], conf=confidence_threshold, device='cuda', verbose=False)

            # Check if any vehicle detected with sufficient confidence
            if len(results[0].boxes) > 0:
                frames_with_vehicles.append(frame_idx)

            frame_idx += 1
            pbar.update(1)

    cap.release()

    if not frames_with_vehicles:
        print("No vehicles detected in the video!")
        return []

    # Identify continuous segments
    segments = []
    start_frame = frames_with_vehicles[0]
    prev_frame = frames_with_vehicles[0]

    for frame in frames_with_vehicles[1:]:
        # If gap > 1 frame, start new segment
        if frame - prev_frame > 1:
            # Save previous segment if it meets minimum duration
            if prev_frame - start_frame + 1 >= min_frames:
                segments.append((start_frame, prev_frame))
            start_frame = frame
        prev_frame = frame

    # Add last segment
    if prev_frame - start_frame + 1 >= min_frames:
        segments.append((start_frame, prev_frame))

    print(f"\nFound {len(segments)} vehicle segments (after filtering)")
    total_vehicle_frames = sum(end - start + 1 for start, end in segments)
    total_vehicle_duration = total_vehicle_frames / fps
    print(f"Total vehicle duration: {total_vehicle_duration:.2f}s ({total_vehicle_frames} frames)")

    return segments, fps

def extract_and_concatenate_segments(video_path, segments, fps, output_path):
    """
    Extract vehicle segments from video and concatenate them into one output video

    Args:
        video_path: Path to input video
        segments: List of (start_frame, end_frame) tuples
        fps: Frames per second of input video
        output_path: Path to save output video
    """
    if not segments:
        print("No segments to extract!")
        return

    # Open input video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nExtracting and concatenating {len(segments)} segments...")

    total_frames = sum(end - start + 1 for start, end in segments)

    with tqdm(total=total_frames, desc="Writing output video") as pbar:
        for seg_idx, (start_frame, end_frame) in enumerate(segments):
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read and write frames in this segment
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                pbar.update(1)

    cap.release()
    out.release()

    print(f"\nOutput video saved to: {output_path}")

def process_video(input_video, output_video=None, confidence_threshold=0.5, min_segment_duration=0.5):
    """
    Main function to process video and extract vehicle segments

    Args:
        input_video: Path to input video file
        output_video: Path to save output video (optional, auto-generated if not provided)
        confidence_threshold: Minimum confidence for vehicle detection (default: 0.5)
        min_segment_duration: Minimum duration in seconds for segments (default: 0.5)
    """
    # Verify input file exists
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Generate output path if not provided
    if output_video is None:
        input_path = Path(input_video)
        output_dir = Path('output/not/extracted_segments')
        output_video = output_dir / f"{input_path.stem}_vehicles_only.mp4"

    print("="*60)
    print("VEHICLE SEGMENT EXTRACTION")
    print("="*60)

    # Step 1: Detect vehicle segments
    segments, fps = detect_vehicle_segments(input_video, confidence_threshold, min_segment_duration)

    if not segments:
        print("\nNo vehicle segments found. Exiting.")
        return None

    # Step 2: Extract and concatenate segments
    extract_and_concatenate_segments(input_video, segments, fps, str(output_video))

    print("="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)

    return str(output_video)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract vehicle segments from video (GPU required)')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default=None, help='Path to output video (optional)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold (0-1)')
    parser.add_argument('--min-duration', type=float, default=0.5, help='Minimum segment duration in seconds')

    args = parser.parse_args()

    try:
        output_path = process_video(
            args.input_video,
            args.output,
            args.confidence,
            args.min_duration
        )

        if output_path:
            print(f"\nSuccess! Output saved to: {output_path}")

    except RuntimeError as e:
        if "GPU not available" in str(e):
            print(f"\n{e}")
            exit(1)
        raise
    except Exception as e:
        print(f"\nError: {e}")
        raise

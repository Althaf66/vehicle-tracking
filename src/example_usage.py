"""
Example usage of the vehicle segment extraction tool
"""

from extract_vehicle_segments import process_video

def example_basic_usage():
    """Basic usage example - auto-generates output path"""
    print("Example 1: Basic Usage")
    print("-" * 50)

    output_path = process_video(
        input_video='data/raw_videos/1.mp4'
    )

    print(f"\nOutput saved to: {output_path}")


def example_custom_parameters():
    """Example with custom parameters"""
    print("\n\nExample 2: Custom Parameters")
    print("-" * 50)

    output_path = process_video(
        input_video='data/raw_videos/1.mp4',
        output_video='output/1_custom_output.mp4',
        confidence_threshold=0.6,  # Higher confidence = fewer false positives
        min_segment_duration=1.0   # Only keep segments >= 1 second
    )

    print(f"\nOutput saved to: {output_path}")


def example_low_confidence():
    """Example with lower confidence to catch more vehicles"""
    print("\n\nExample 3: Lower Confidence (More Detections)")
    print("-" * 50)

    output_path = process_video(
        input_video='data/raw_videos/1.mp4',
        output_video='output/more_vehicles.mp4',
        confidence_threshold=0.3,  # Lower confidence = more detections
        min_segment_duration=0.3   # Shorter segments allowed
    )

    print(f"\nOutput saved to: {output_path}")


def example_strict_filtering():
    """Example with strict filtering"""
    print("\n\nExample 4: Strict Filtering")
    print("-" * 50)

    output_path = process_video(
        input_video='data/raw_videos/1.mp4',
        output_video='output/strict_filtered.mp4',
        confidence_threshold=0.7,  # High confidence only
        min_segment_duration=2.0   # Only segments >= 2 seconds
    )

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    # Uncomment the example you want to run

    # example_basic_usage()
    example_custom_parameters()
    # example_low_confidence()
    # example_strict_filtering()

    print("\nPlease uncomment one of the example functions to run it.")
    print("Make sure to replace 'your_video.mp4' with your actual video file.")

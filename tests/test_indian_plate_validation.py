"""
Unit tests for Indian plate format validation and character correction
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.license_plate_recognizer import LicensePlateRecognizer


def test_valid_indian_plates():
    """Test that valid Indian plates are recognized"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)

    test_cases = [
        ("KL01AA3456", True),  # Standard 10-char format
        ("KL01A3456", True),   # 9-char format
        ("MH12DE1234", True),  # Different state
        ("DL01AB1234", True),  # Delhi
        ("TN09BC5678", True),  # Tamil Nadu
    ]

    print("Testing Valid Indian Plates:")
    print("=" * 60)

    for plate, expected in test_cases:
        corrected, is_valid = recognizer.validate_indian_plate_format(plate)
        status = "[PASS]" if is_valid == expected else "[FAIL]"
        print(f"{status} | Input: {plate} | Valid: {is_valid} | Output: {corrected}")

    print()


def test_character_confusion_correction():
    """Test character confusion correction (O<->0, I<->1, B<->8, etc.)"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)

    test_cases = [
        # (input, expected_output, should_be_valid)
        ("KLO1AA3456", "KL01AA3456", True),  # O->0 in position 2
        ("KL0IAA3456", "KL01AA3456", True),  # I->1 in position 3
        ("KL01883456", "KL01BB3456", True),  # 8->B in position 4-5
        ("KL01AA345S", "KL01AA3455", True),  # S->5 in last position
        ("0L01AA3456", "OL01AA3456", True),  # 0->O in position 0
        ("1L01AA3456", "IL01AA3456", True),  # 1->I in position 0
        ("KL01AA34S6", "KL01AA3456", True),  # S->5 in digit section
    ]

    print("Testing Character Confusion Correction:")
    print("=" * 60)

    for input_plate, expected_output, should_be_valid in test_cases:
        corrected, is_valid = recognizer.validate_indian_plate_format(input_plate)

        output_match = corrected == expected_output
        valid_match = is_valid == should_be_valid

        status = "[PASS]" if (output_match and valid_match) else "[FAIL]"
        print(f"{status} | Input: {input_plate} | Expected: {expected_output} | "
              f"Got: {corrected} | Valid: {is_valid}")

    print()


def test_invalid_plates():
    """Test that invalid plates are rejected"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)

    test_cases = [
        "ABC123",        # Too short
        "ABCD1234567",   # Too long
        "1234567890",    # All digits
        "ABCDEFGHIJ",    # All letters
        "KL01",          # Too short
        "RANDOMTEXT",    # Random text
        "KL01AA34",      # Missing digits
    ]

    print("Testing Invalid Plates (Should be Rejected):")
    print("=" * 60)

    for plate in test_cases:
        corrected, is_valid = recognizer.validate_indian_plate_format(plate)
        status = "[PASS]" if not is_valid else "[FAIL] (Should be invalid)"
        print(f"{status} | Input: {plate} | Valid: {is_valid} | Output: {corrected}")

    print()


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    recognizer = LicensePlateRecognizer(gpu_enabled=False)

    test_cases = [
        # Plates with spaces (should be cleaned)
        ("KL 01 AA 3456", "KL01AA3456", True),
        ("KL-01-AA-3456", "KL01AA3456", True),

        # Mixed case (should be uppercased)
        ("kl01aa3456", "KL01AA3456", True),
        ("Kl01Aa3456", "KL01AA3456", True),

        # With special characters
        ("KL@01#AA$3456", "KL01AA3456", True),
    ]

    print("Testing Edge Cases:")
    print("=" * 60)

    for input_plate, expected_output, should_be_valid in test_cases:
        corrected, is_valid = recognizer.validate_indian_plate_format(input_plate)

        output_match = corrected == expected_output
        valid_match = is_valid == should_be_valid

        status = "[PASS]" if (output_match and valid_match) else "[FAIL]"
        print(f"{status} | Input: '{input_plate}' | Expected: {expected_output} | "
              f"Got: {corrected} | Valid: {is_valid}")

    print()


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "=" * 70)
    print("INDIAN LICENSE PLATE VALIDATION TEST SUITE")
    print("=" * 70 + "\n")

    test_valid_indian_plates()
    test_character_confusion_correction()
    test_invalid_plates()
    test_edge_cases()

    print("=" * 70)
    print("Test suite completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

import os
import shutil
from sklearn.model_selection import train_test_split


def split_val_to_test(data_dir, test_size=0.5):
    """Split validation data into val and test sets.

    Moves a portion of images from val/ to test/ for each class,
    creating a truly unseen test set.

    Args:
        data_dir: Root directory containing val/ subdirectory
        test_size: Fraction of val images to move to test (default 0.5)
    """
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(val_dir):
        print(f"ERROR: Validation directory not found: {val_dir}")
        return

    print(f"\nSplitting val data in: {data_dir}")
    print(f"Test size: {test_size:.0%} of validation set")
    print("-" * 50)
    print(f"{'Class':<25} {'Val (remaining)':<18} {'Test (new)':<12}")
    print("-" * 50)

    for class_name in sorted(os.listdir(val_dir)):
        class_val_path = os.path.join(val_dir, class_name)

        if not os.path.isdir(class_val_path):
            continue

        images = [f for f in os.listdir(class_val_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if len(images) < 2:
            print(f"{class_name:<25} {'skipped (too few images)'}")
            continue

        # Split: keep some in val, move rest to test
        val_imgs, test_imgs = train_test_split(
            images, test_size=test_size, random_state=42
        )

        # Create test class directory
        class_test_path = os.path.join(test_dir, class_name)
        os.makedirs(class_test_path, exist_ok=True)

        # Move selected images to test
        for img in test_imgs:
            src = os.path.join(class_val_path, img)
            dst = os.path.join(class_test_path, img)
            shutil.move(src, dst)

        print(f"{class_name:<25} {len(val_imgs):<18} {len(test_imgs):<12}")

    print("-" * 50)
    print("Done!\n")


if __name__ == '__main__':
    print("=" * 50)
    print("Preparing Test Data (from Validation Set)")
    print("=" * 50)

    split_val_to_test('data/training_data/color')
    split_val_to_test('data/training_data/car_name')

    print("=" * 50)
    print("Test data preparation complete!")
    print("=" * 50)

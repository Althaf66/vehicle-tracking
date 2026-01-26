import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import shutil
from tqdm import tqdm

class VehicleCarNameCategorizer:
    def __init__(self, model_path='models/car_name_classifier.pth',
                 classes_path='models/carname_classes.json',
                 input_dir='data/extracted_vehicles/entrance',
                 output_base_dir='data/categorized_by_carname/entrance'):
        """
        Initialize the vehicle car name categorizer.

        Args:
            model_path: Path to the trained car name classifier model
            classes_path: Path to the JSON file containing car name class names
            input_dir: Directory containing extracted vehicle images
            output_base_dir: Base directory where categorized images will be saved
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.input_dir = input_dir
        self.output_base_dir = output_base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load class names
        with open(self.classes_path, 'r') as f:
            self.class_names = json.load(f)

        print(f"Loaded {len(self.class_names)} car name classes")

        # Load model
        self.model = self._load_model()

        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create output directories for each car name
        self._create_output_directories()

    def _load_model(self):
        """Load the trained car name classifier model."""
        num_classes = len(self.class_names)
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        # Load trained weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        print(f"Model loaded from {self.model_path}")
        print(f"Using device: {self.device}")
        return model

    def _create_output_directories(self):
        """Create output directories for each car name class and 'other' folder."""
        os.makedirs(self.output_base_dir, exist_ok=True)

        # Create folders for each car model
        for car_name in self.class_names:
            car_dir = os.path.join(self.output_base_dir, car_name)
            os.makedirs(car_dir, exist_ok=True)

        # Create 'other' folder for low-confidence and error cases
        other_dir = os.path.join(self.output_base_dir, 'other')
        os.makedirs(other_dir, exist_ok=True)

        print(f"Created output directories in {self.output_base_dir}")
        print(f"Created 'other' folder for low-confidence/error cases")

    def predict_car_name(self, image_path):
        """
        Predict the car name of a vehicle image.

        Args:
            image_path: Path to the vehicle image

        Returns:
            Tuple of (predicted_car_name, confidence_score)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_car_name = self.class_names[predicted.item()]
            confidence_score = confidence.item()

            return predicted_car_name, confidence_score

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, 0.0

    def categorize_all_images(self, copy_files=True, min_confidence=0.5):
        """
        Categorize all vehicle images by car name.

        Args:
            copy_files: If True, copy files; if False, move files
            min_confidence: Minimum confidence threshold (0.0 to 1.0) to categorize an image
        """
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in os.listdir(self.input_dir)
                      if os.path.isfile(os.path.join(self.input_dir, f)) and
                      os.path.splitext(f.lower())[1] in image_extensions]

        print(f"\nFound {len(image_files)} images to categorize")
        print(f"{'Copying' if copy_files else 'Moving'} files to car name folders...")
        print(f"Minimum confidence threshold: {min_confidence:.2f}\n")

        # Statistics
        carname_counts = {car_name: 0 for car_name in self.class_names}
        low_confidence_count = 0
        error_count = 0
        other_count = 0

        # Process each image
        for image_file in tqdm(image_files, desc="Categorizing images"):
            image_path = os.path.join(self.input_dir, image_file)

            # Predict car name
            predicted_car_name, confidence = self.predict_car_name(image_path)

            # Determine destination folder
            if predicted_car_name is None:
                # Error case - move to 'other'
                dest_dir = os.path.join(self.output_base_dir, 'other')
                error_count += 1
                other_count += 1
            elif confidence < min_confidence:
                # Low confidence - move to 'other'
                dest_dir = os.path.join(self.output_base_dir, 'other')
                low_confidence_count += 1
                other_count += 1
            else:
                # High confidence - move to predicted car name folder
                dest_dir = os.path.join(self.output_base_dir, predicted_car_name)
                carname_counts[predicted_car_name] += 1

            dest_path = os.path.join(dest_dir, image_file)

            try:
                if copy_files:
                    shutil.copy2(image_path, dest_path)
                else:
                    shutil.move(image_path, dest_path)

            except Exception as e:
                print(f"\nError copying/moving {image_file}: {str(e)}")
                error_count += 1

        # Print summary
        self._print_summary(carname_counts, low_confidence_count, error_count,
                          other_count, len(image_files))

    def _print_summary(self, carname_counts, low_confidence_count, error_count,
                      other_count, total_images):
        """Print categorization summary."""
        print("\n" + "="*60)
        print("CATEGORIZATION SUMMARY")
        print("="*60)

        successfully_categorized = sum(carname_counts.values())

        print(f"\nTotal images processed: {total_images}")
        print(f"Successfully categorized: {successfully_categorized}")
        print(f"Low confidence (moved to 'other'): {low_confidence_count}")
        print(f"Errors (moved to 'other'): {error_count}")
        print(f"Total in 'other' folder: {other_count}")

        print("\nImages per car name:")
        print("-" * 40)
        for car_name, count in sorted(carname_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:  # Only show car names with at least 1 image
                percentage = (count / successfully_categorized * 100) if successfully_categorized > 0 else 0
                print(f"  {car_name:30s}: {count:5d} ({percentage:5.1f}%)")

        print("\n" + "="*60)
        print(f"Output directory: {self.output_base_dir}")
        print("="*60 + "\n")


def main():
    """Main function to run the categorization."""
    # Configuration
    # Available directories: entrance (701), entrance (5711), Phase1 (5), entrance (1002)
    MODEL_PATH = 'models/car_name_classifier.pth'
    CLASSES_PATH = 'models/carname_classes.json'
    INPUT_DIR = 'data/extracted_vehicles/entrance'  # Change to: entrance, entrance, Phase1, or entrance
    OUTPUT_DIR = 'data/categorized_by_carname/entrance'

    # Create categorizer instance
    categorizer = VehicleCarNameCategorizer(
        model_path=MODEL_PATH,
        classes_path=CLASSES_PATH,
        input_dir=INPUT_DIR,
        output_base_dir=OUTPUT_DIR
    )

    # Categorize all images
    # Set copy_files=True to copy (keeps originals), False to move
    # Set min_confidence to filter low-confidence predictions (e.g., 0.5 for 50%)
    categorizer.categorize_all_images(
        copy_files=True,  # Copy files (keep originals)
        min_confidence=0.5  # Images below 50% confidence go to 'other' folder
    )


if __name__ == '__main__':
    main()

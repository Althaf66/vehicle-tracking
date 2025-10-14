import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, train_ratio=0.8):
    """Split data into train and validation sets"""
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        train_imgs, val_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        
        train_class_dir = os.path.join(source_dir, 'train', class_name)
        val_class_dir = os.path.join(source_dir, 'val', class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy(src, dst)
        
        for img in val_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy(src, dst)
        
        print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

split_dataset('data/training_data/color')
split_dataset('data/training_data/car_name')
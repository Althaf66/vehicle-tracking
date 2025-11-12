# Multi-Camera Vehicle Tracking System

This system assigns unique IDs to vehicles based on their color and car name. The same vehicle will maintain the same ID across multiple cameras.

## Step 1: Setup Environment
Create virtual environment and install required libraries
```
python -m venv vehicle_tracking
- On Linux: source vehicle_tracking/bin/activate  
- On Windows: vehicle_tracking\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Extract vehicle from raw video

Copy raw footage videos on `src/data/raw_videos` and run 
```
cd src
python extract_vehicles.py
```
This will extract vehicles from raw footage and save them in `src/data/extracted_vehicles` folder.

## Step 3: Manually organize images into folders

Manually copy extracted vehicles to `src/data/training_data` with their particular named folder inside color and car_name. 
Eg:- image of white Celerio should be copy in both `src/data/training_data/color/white` and `src/data/training_data/car_name/marutisuzuki_celerio`.

## Step 4: Prepare training/validation split:
```
python prepare_training_data.py
```

## Step 5: Train the model car name and color classifier:
```
python train_color_classifier.py
python train_carname_classifier.py
```

## Step 6: Run camera tracking:
To get preview output in the window run
```
python preview_output.py
```
Output video shoud be saved in `output/tracked_videos/` and excel file saved on `output/report/`




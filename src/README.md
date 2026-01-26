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

### Option A: Extract Vehicle Segments (NEW - GPU Required)
Extract and concatenate video segments containing only vehicles from long footage:
```
cd src
python extract_vehicle_segments.py path/to/your/video.mp4
```
This creates a single video showing only the parts where vehicles are visible. See `src/VEHICLE_SEGMENT_EXTRACTION_GUIDE.md` for detailed usage.

### Option B: Extract Vehicle Images
Copy raw footage videos on `src/data/raw_videos` and run
```
cd src
python extract_vehicles.py
```
This will extract vehicles from raw footage and save them in `src/data/extracted_vehicles` folder.

## Step 3: organize images into folders

### Manually 

Manually copy extracted vehicles to `src/data/training_data` with their particular named folder inside color and car_name. 
Eg:- image of white Celerio should be copy in both `src/data/training_data/color/white` and `src/data/training_data/car_name/marutisuzuki_celerio`.

### With help of model

Use trained models to automatically categorize extracted vehicle images:

**Categorize by Color:**
```
python categorize_vehicles_by_color.py
```
Organizes images into color folders (black, blue, grey, maroon, red, silver, white, yellow).

**Categorize by Car Name:**
```
python categorize_vehicles_by_carname.py
```
Organizes images into 73 car model folders (e.g., honda_city, maruti_suzuki_swift, tata_nexon). Low-confidence predictions go to 'other' folder.

Both scripts:
- Copy files (keep originals)
- Support confidence threshold filtering
- Output to `src/data/categorized_by_color/` or `src/data/categorized_by_carname/`


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




import json
import numpy as np

class VehicleIDGenerator:
    def __init__(self, color_confidence_threshold=0.6, carname_confidence_threshold=0.7):
        # Load class mappings
        with open('models/color_classes.json', 'r') as f:
            self.color_classes = json.load(f)
        
        with open('models/carname_classes.json', 'r') as f:
            self.carname_classes = json.load(f)
        
        # Create ID mapping
        self.color_codes = {color: (i+1)*100 for i, color in enumerate(self.color_classes)}
        self.carname_codes = {name: i+1 for i, name in enumerate(self.carname_classes)}
        
        # Confidence thresholds
        self.color_conf_threshold = color_confidence_threshold
        self.carname_conf_threshold = carname_confidence_threshold
    
    def generate_id(self, color, car_name, color_conf, carname_conf):
        """
        Generate unique ID with confidence validation
        Returns: (vehicle_id, is_valid)
        """
        # Check confidence thresholds
        if color_conf < self.color_conf_threshold or carname_conf < self.carname_conf_threshold:
            return None, False  # Low confidence, reject prediction
        
        color_code = self.color_codes.get(color, 0)
        carname_code = self.carname_codes.get(car_name, 0)
        
        vehicle_id = color_code + carname_code
        return vehicle_id, True
    
    def get_attributes_from_id(self, vehicle_id):
        """Reverse: Get color and car name from ID"""
        if vehicle_id is None:
            return None, None
        
        color_code = (vehicle_id // 100) * 100
        carname_code = vehicle_id % 100
        
        color = [k for k, v in self.color_codes.items() if v == color_code][0]
        car_name = [k for k, v in self.carname_codes.items() if v == carname_code][0]
        
        return color, car_name
"""
Scene Classification using Places365
=====================================

Classifies images into 365 scene categories (beach, mountain, museum, etc.)
Uses pretrained ResNet50 model fine-tuned on Places365 dataset.

Author: Evan Petersen
Date: November 2025
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import requests
from PIL import Image


# ============================================================================
# SCENE CATEGORIES MAPPING
# ============================================================================

# Complete Places365 scene categories (365 total)
SCENE_CATEGORIES = [
    'airfield', 'airplane_cabin', 'airport_terminal', 'alcove', 'alley',
    'amphitheater', 'amusement_arcade', 'amusement_park', 'apartment_building_outdoor', 'aquarium',
    'aqueduct', 'arch', 'archaelogical_excavation', 'archive', 'arena_hockey',
    'arena_performance', 'arena_rodeo', 'army_base', 'art_gallery', 'art_school',
    'art_studio', 'artists_loft', 'assembly_line', 'athletic_field_outdoor', 'atrium_public',
    'attic', 'auditorium', 'auto_factory', 'auto_showroom', 'badlands',
    'bakery_shop', 'balcony_exterior', 'balcony_interior', 'ball_pit', 'ballroom',
    'bamboo_forest', 'bank_vault', 'banquet_hall', 'bar', 'barn',
    'barndoor', 'baseball_field', 'basement', 'basketball_court_indoor', 'bathroom',
    'bazaar_indoor', 'bazaar_outdoor', 'beach', 'beach_house', 'beauty_salon',
    'bedchamber', 'bedroom', 'beer_garden', 'beer_hall', 'berth',
    'biology_laboratory', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore',
    'booth_indoor', 'botanical_garden', 'bow_window_indoor', 'bowling_alley', 'boxing_ring',
    'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior',
    'bus_station_indoor', 'butchers_shop', 'butte', 'cabin_outdoor', 'cafeteria',
    'campsite', 'campus', 'canal_natural', 'canal_urban', 'candy_store',
    'canyon', 'car_interior', 'carrousel', 'castle', 'catacomb',
    'cathedral_exterior', 'cathedral_interior', 'cavern_indoor', 'cemetery', 'chalet',
    'chemistry_lab', 'childs_room', 'church_indoor', 'church_outdoor', 'classroom',
    'clean_room', 'cliff', 'closet', 'clothing_store', 'coast',
    'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room',
    'construction_site', 'corn_field', 'corral', 'corridor', 'cottage',
    'courthouse', 'courtyard', 'crevasse', 'crosswalk', 'dam',
    'delicatessen', 'department_store', 'desert_sand', 'desert_vegetation', 'diner_outdoor',
    'dining_hall', 'dining_room', 'discotheque', 'doorway_outdoor', 'dorm_room',
    'downtown', 'dressing_room', 'driveway', 'drugstore', 'embassy',
    'engine_room', 'entrance_hall', 'escalator_indoor', 'excavation', 'fabric_store',
    'farm', 'fastfood_restaurant', 'field_cultivated', 'field_wild', 'fire_escape',
    'fire_station', 'fishpond', 'flea_market_indoor', 'florist_shop_indoor', 'food_court',
    'football_field', 'forest_broadleaf', 'forest_path', 'forest_road', 'formal_garden',
    'fountain', 'galley', 'garage_indoor', 'garage_outdoor', 'gas_station',
    'gazebo_exterior', 'general_store_indoor', 'general_store_outdoor', 'gift_shop', 'glacier',
    'golf_course', 'greenhouse_indoor', 'greenhouse_outdoor', 'grotto', 'gymnasium_indoor',
    'hangar_indoor', 'hangar_outdoor', 'harbor', 'hardware_store', 'hayfield',
    'heliport', 'highway', 'home_office', 'home_theater', 'hospital',
    'hospital_room', 'hot_spring', 'hotel_outdoor', 'hotel_room', 'house',
    'hunting_lodge_outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink_indoor',
    'ice_skating_rink_outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn_outdoor',
    'islet', 'jacuzzi_indoor', 'jail_cell', 'japanese_garden', 'jewelry_shop',
    'junkyard', 'kasbah', 'kennel_outdoor', 'kindergarden_classroom', 'kitchen',
    'lagoon', 'lake_natural', 'landfill', 'landing_deck', 'laundromat',
    'lawn', 'lecture_room', 'legislative_chamber', 'library_indoor', 'library_outdoor',
    'lighthouse', 'living_room', 'loading_dock', 'lobby', 'lock_chamber',
    'locker_room', 'mansion', 'manufactured_home', 'market_indoor', 'market_outdoor',
    'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'mezzanine',
    'moat_water', 'mosque_outdoor', 'motel', 'mountain', 'mountain_path',
    'mountain_snowy', 'movie_theater_indoor', 'museum_indoor', 'museum_outdoor', 'music_studio',
    'natural_history_museum', 'nursery', 'nursing_home', 'oast_house', 'ocean',
    'office', 'office_building', 'office_cubicles', 'oilrig', 'operating_room',
    'orchard', 'orchestra_pit', 'pagoda', 'palace', 'pantry',
    'park', 'parking_garage_indoor', 'parking_garage_outdoor', 'parking_lot', 'pasture',
    'patio', 'pavilion', 'pet_shop', 'pharmacy', 'phone_booth',
    'physics_laboratory', 'picnic_area', 'pier', 'pizzeria', 'playground',
    'playroom', 'plaza', 'pond', 'porch', 'promenade',
    'pub_indoor', 'pulpit', 'putting_green', 'racecourse', 'raceway',
    'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room',
    'repair_shop', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio',
    'rice_paddy', 'river', 'rock_arch', 'roof_garden', 'rope_bridge',
    'ruin', 'runway', 'sandbox', 'sauna', 'schoolhouse',
    'science_museum', 'server_room', 'shed', 'shoe_shop', 'shopfront',
    'shopping_mall_indoor', 'shower', 'ski_resort', 'ski_slope', 'sky',
    'skyscraper', 'slum', 'snowfield', 'soccer_field', 'stable',
    'stadium_baseball', 'stadium_football', 'stadium_soccer', 'stage_indoor', 'stage_outdoor',
    'staircase', 'storage_room', 'street', 'subway_station_platform', 'supermarket',
    'sushi_bar', 'swamp', 'swimming_hole', 'swimming_pool_indoor', 'swimming_pool_outdoor',
    'synagogue_outdoor', 'television_room', 'television_studio', 'temple_asia', 'throne_room',
    'ticket_booth', 'topiary_garden', 'tower', 'toyshop', 'train_interior',
    'train_station_platform', 'tree_farm', 'tree_house', 'trench', 'tundra',
    'underwater_ocean_deep', 'utility_room', 'valley', 'vegetable_garden', 'veranda',
    'veterinarians_office', 'viaduct', 'village', 'vineyard', 'volcano',
    'volleyball_court_outdoor', 'waiting_room', 'water_park', 'water_tower', 'waterfall',
    'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm',
    'windmill', 'yard', 'youth_hostel', 'zen_garden'
]


# ============================================================================
# SCENE CLASSIFIER CLASS
# ============================================================================

class SceneClassifier:
    """
    Scene classification using pretrained Places365 model.
    
    Attributes:
        model: PyTorch model (ResNet50 trained on Places365)
        device: Computation device (cuda or cpu)
        categories: List of scene category names
        transform: Image preprocessing pipeline
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 num_classes: int = 365):
        """
        Initialize the scene classifier.
        
        Args:
            model_path: Path to pretrained model weights (optional)
            device: Device to run inference on ('cuda' or 'cpu')
            num_classes: Number of scene categories (default 365)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load category names
        self._load_categories()
        
        # Initialize model
        self.model = self._build_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = self._get_transform()
        
        print(f"SceneClassifier initialized on {self.device}")
    
    
    def _build_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Build and load the Places365 model.
        
        Args:
            model_path: Path to pretrained weights
            
        Returns:
            Loaded PyTorch model
        """
        # Use ResNet50 architecture
        model = models.resnet50(pretrained=False)
        
        # Modify final layer for Places365
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Load pretrained Places365 weights if available
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print("Warning: No pretrained weights loaded. Using ImageNet initialization.")
            print("For best results, download Places365 weights:")
            print("http://places2.csail.mit.edu/models_places365/")
        
        return model
    
    
    def _get_transform(self) -> transforms.Compose:
        """
        Get preprocessing transform for Places365 model.
        
        Returns:
            Composed transforms
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    
    def _load_categories(self):
        """Load scene category names from default list."""
        self.categories = SCENE_CATEGORIES[:self.num_classes]
    
    
    # ========================================================================
    # PREDICTION METHODS
    # ========================================================================
    
    def predict(self, 
                image: Image.Image, 
                top_k: int = 5) -> List[Dict[str, any]]:
        """
        Predict scene categories for an image.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of dicts with 'category', 'confidence', and 'index'
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'category': self.categories[idx.item()],
                'confidence': prob.item(),
                'index': idx.item()
            })
        
        return predictions
    
    
    def predict_batch(self, 
                     images: List[Image.Image], 
                     top_k: int = 5) -> List[List[Dict[str, any]]]:
        """
        Predict scene categories for multiple images.
        
        Args:
            images: List of PIL Images
            top_k: Number of top predictions per image
            
        Returns:
            List of prediction lists for each image
        """
        # Preprocess all images
        img_tensors = torch.stack([
            self.transform(img) for img in images
        ]).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensors)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k for each image
        results = []
        for probs in probabilities:
            top_probs, top_indices = torch.topk(probs, top_k)
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'category': self.categories[idx.item()],
                    'confidence': prob.item(),
                    'index': idx.item()
                })
            results.append(predictions)
        
        return results
    
    
    def get_top_category(self, image: Image.Image) -> str:
        """
        Get the single most likely scene category.
        
        Args:
            image: PIL Image
            
        Returns:
            Category name as string
        """
        predictions = self.predict(image, top_k=1)
        return predictions[0]['category']
    
    
    def get_all_predictions(self, image: Image.Image) -> Dict[str, float]:
        """
        Get probabilities for all scene categories.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary mapping category names to probabilities
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        return {
            self.categories[i]: prob.item() 
            for i, prob in enumerate(probabilities)
        }
    
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def filter_by_category_type(self, 
                                predictions: List[Dict[str, any]], 
                                category_type: str) -> List[Dict[str, any]]:
        """
        Filter predictions by category type (e.g., outdoor, indoor, natural).
        
        Args:
            predictions: List of prediction dicts
            category_type: Type to filter by
            
        Returns:
            Filtered list of predictions
        """
        # Define category types
        outdoor_scenes = {'beach', 'mountain', 'forest', 'desert', 'field', 'park', 
                         'garden', 'plaza', 'street', 'bridge', 'canal'}
        indoor_scenes = {'museum', 'church', 'cathedral', 'palace', 'castle', 
                        'restaurant', 'cafe', 'hotel', 'lobby', 'gallery'}
        natural_scenes = {'beach', 'mountain', 'waterfall', 'forest', 'jungle',
                         'lake', 'river', 'ocean', 'cliff', 'canyon'}
        
        category_map = {
            'outdoor': outdoor_scenes,
            'indoor': indoor_scenes,
            'natural': natural_scenes
        }
        
        if category_type not in category_map:
            return predictions
        
        relevant_categories = category_map[category_type]
        
        return [
            pred for pred in predictions 
            if any(keyword in pred['category'] for keyword in relevant_categories)
        ]
    
    
    def is_travel_relevant(self, predictions: List[Dict[str, any]], 
                          threshold: float = 0.3) -> bool:
        """
        Check if image is travel/location relevant based on scene classification.
        
        Args:
            predictions: List of prediction dicts
            threshold: Minimum confidence threshold
            
        Returns:
            True if travel-relevant scene detected
        """
        travel_scenes = {
            'beach', 'mountain', 'castle', 'palace', 'temple', 'church',
            'cathedral', 'mosque', 'pagoda', 'museum', 'gallery', 'plaza',
            'square', 'bridge', 'tower', 'monument', 'waterfall', 'canyon',
            'lake', 'ocean', 'harbor', 'lighthouse', 'stadium', 'arena'
        }
        
        for pred in predictions:
            if pred['confidence'] >= threshold:
                if any(keyword in pred['category'].lower() for keyword in travel_scenes):
                    return True
        
        return False
    
    
    def __repr__(self) -> str:
        """String representation of the classifier."""
        return (f"SceneClassifier(device='{self.device}', "
                f"num_classes={self.num_classes}, "
                f"categories={len(self.categories)})")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def download_places365_weights(save_path: str = "models/weights/"):
    """
    Download pretrained Places365 weights.
    
    Args:
        save_path: Directory to save weights
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
    output_file = Path(save_path) / "resnet50_places365.pth"
    
    if output_file.exists():
        print(f"Weights already exist at {output_file}")
        return
    
    print(f"Downloading Places365 weights from {url}")
    print("This may take several minutes...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_file, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}%", end='')
    
    print(f"\nWeights saved to {output_file}")


def load_scene_categories(file_path: str) -> List[str]:
    """
    Load scene category names from file.
    
    Args:
        file_path: Path to categories file
        
    Returns:
        List of category names
    """
    with open(file_path, 'r') as f:
        categories = [line.strip().split(' ')[0] for line in f.readlines()]
    return categories

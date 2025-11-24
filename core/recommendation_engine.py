"""
Location Recommendation Engine
Finds landmarks near user's itinerary based on feature similarity.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from scipy.spatial import cKDTree
import math


@dataclass
class Recommendation:
    """A single landmark recommendation."""
    name: str
    landmark_id: int
    latitude: float
    longitude: float
    distance_km: float
    similarity_score: float
    final_score: float
    country: str
    description: str
    closest_itinerary_item: str


class RecommendationEngine:
    """
    Recommends landmarks based on proximity and feature similarity.
    """
    
    def __init__(self, landmarks_path: str = None):
        """
        Initialize recommendation engine.
        
        Args:
            landmarks_path: Path to enriched landmarks JSON
        """
        if landmarks_path is None:
            landmarks_path = Path(__file__).parent.parent / 'data' / 'landmarks_unified.json'
        
        # Load landmarks database
        with open(landmarks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.landmarks = data['landmarks']
        
        # Filter out landmarks without coordinates
        self.landmarks = [lm for lm in self.landmarks if 'latitude' in lm]
        
        print(f"Loaded {len(self.landmarks)} landmarks with coordinates")
        
        # Build spatial index for fast proximity search
        print("Building spatial index...")
        self._build_spatial_index()
        print(f"[OK] Spatial index ready ({len(self.landmarks)} locations)")
        
        # Load sentence transformer for feature embeddings
        print("Loading sentence transformer model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, fast
        print("[OK] Sentence transformer ready")
        
        # Pre-compute embeddings for all landmarks (optional, speeds up queries)
        self._precompute_embeddings()
        
        # Load CLIP visual embeddings if available
        self._load_clip_embeddings()
    
    def _build_spatial_index(self):
        """Build KD-Tree for fast spatial queries."""
        if not self.landmarks:
            self.spatial_index = None
            self.landmark_coords = []
            return
        
        # Extract coordinates (lat, lon)
        self.landmark_coords = np.array([
            [lm['latitude'], lm['longitude']] 
            for lm in self.landmarks
        ])
        
        # Build KD-Tree (logarithmic search time)
        self.spatial_index = cKDTree(self.landmark_coords)
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for all landmark descriptions."""
        print("Pre-computing landmark embeddings...")
        
        descriptions = []
        for lm in self.landmarks:
            desc = lm.get('description', lm['name'])
            descriptions.append(desc)
        
        # Batch embed all descriptions (much faster than one-by-one)
        self.landmark_embeddings = self.embedder.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"[OK] Pre-computed {len(self.landmark_embeddings)} embeddings")
    
    def _load_clip_embeddings(self):
        """Load pre-computed CLIP visual embeddings."""
        embeddings_path = Path(__file__).parent.parent / 'data' / 'landmarks_clip_embeddings.npy'
        mapping_path = Path(__file__).parent.parent / 'data' / 'landmarks_id_mapping.json'
        
        if not embeddings_path.exists() or not mapping_path.exists():
            print("[INFO] No CLIP visual embeddings found (visual similarity disabled)")
            self.clip_embeddings = None
            self.clip_id_to_idx = {}
            return
        
        print("Loading CLIP visual embeddings...")
        
        # Load embeddings array
        self.clip_embeddings = np.load(embeddings_path)
        
        # Load landmark ID mapping
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            landmark_ids = mapping_data['landmark_ids']
        
        # Create lookup dict: landmark_id -> embedding index
        self.clip_id_to_idx = {lid: idx for idx, lid in enumerate(landmark_ids)}
        
        print(f"[OK] Loaded {len(self.clip_embeddings)} CLIP embeddings")
    
    def haversine_distance(
        self, 
        lat1: float, lon1: float, 
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two points on Earth (in kilometers).
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def get_itinerary_center(
        self, 
        itinerary_landmarks: List[str]
    ) -> Tuple[float, float]:
        """
        Calculate center point of itinerary landmarks.
        
        Args:
            itinerary_landmarks: List of landmark names in itinerary
            
        Returns:
            (latitude, longitude) of center point
        """
        coords = []
        
        for name in itinerary_landmarks:
            landmark = self._find_landmark_by_name(name)
            if landmark and 'latitude' in landmark:
                coords.append((landmark['latitude'], landmark['longitude']))
        
        if not coords:
            raise ValueError("No valid landmarks found in itinerary")
        
        avg_lat = sum(c[0] for c in coords) / len(coords)
        avg_lon = sum(c[1] for c in coords) / len(coords)
        
        return avg_lat, avg_lon
    
    def _find_landmark_by_name(self, name: str) -> Dict[str, Any]:
        """Find landmark by name (case-insensitive)."""
        name_lower = name.lower()
        for landmark in self.landmarks:
            if landmark['name'].lower() == name_lower:
                return landmark
        return None
    
    def find_nearby_landmarks(
        self,
        itinerary_landmarks: List[str],
        max_distance_km: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Find landmarks within radius of itinerary center using spatial index.
        
        Args:
            itinerary_landmarks: List of landmark names
            max_distance_km: Maximum distance from center
            
        Returns:
            List of nearby landmarks with distances
        """
        center_lat, center_lon = self.get_itinerary_center(itinerary_landmarks)
        
        # Use spatial index for fast proximity search
        # Convert km to degrees (approximate: 1 degree ≈ 111 km)
        radius_degrees = max_distance_km / 111.0
        
        if self.spatial_index:
            # Query spatial index (logarithmic time)
            indices = self.spatial_index.query_ball_point(
                [center_lat, center_lon],
                radius_degrees
            )
        else:
            # Fallback: check all landmarks
            indices = range(len(self.landmarks))
        
        nearby = []
        itinerary_names_lower = [name.lower() for name in itinerary_landmarks]
        
        for idx in indices:
            landmark = self.landmarks[idx]
            
            # Skip landmarks already in itinerary
            if landmark['name'].lower() in itinerary_names_lower:
                continue
            
            # Calculate precise distance from center
            dist = self.haversine_distance(
                center_lat, center_lon,
                landmark['latitude'], landmark['longitude']
            )
            
            if dist <= max_distance_km:
                landmark_copy = landmark.copy()
                landmark_copy['distance_km'] = dist
                landmark_copy['_index'] = idx  # Store index for embedding lookup
                
                # Also find closest itinerary item
                min_dist = float('inf')
                closest_item = None
                
                for itin_name in itinerary_landmarks:
                    itin_lm = self._find_landmark_by_name(itin_name)
                    if itin_lm and 'latitude' in itin_lm:
                        d = self.haversine_distance(
                            landmark['latitude'], landmark['longitude'],
                            itin_lm['latitude'], itin_lm['longitude']
                        )
                        if d < min_dist:
                            min_dist = d
                            closest_item = itin_name
                
                landmark_copy['closest_itinerary_item'] = closest_item
                landmark_copy['distance_to_closest'] = min_dist
                
                nearby.append(landmark_copy)
        
        return nearby
    
    def compute_similarity(
        self, 
        llava_description: str, 
        landmark_idx: Optional[int] = None,
        landmark_description: Optional[str] = None
    ) -> float:
        """
        Compute semantic similarity using pre-computed embeddings when available.
        
        Args:
            llava_description: Features from LLaVA
            landmark_idx: Index of landmark (for pre-computed embedding lookup)
            landmark_description: Landmark's description (fallback)
            
        Returns:
            Similarity score (0-1)
        """
        # Embed user's LLaVA description
        llava_embedding = self.embedder.encode([llava_description])[0]
        
        # Get landmark embedding (pre-computed or compute now)
        if landmark_idx is not None and hasattr(self, 'landmark_embeddings'):
            landmark_embedding = self.landmark_embeddings[landmark_idx]
        elif landmark_description:
            landmark_embedding = self.embedder.encode([landmark_description])[0]
        else:
            raise ValueError("Must provide either landmark_idx or landmark_description")
        
        # Cosine similarity
        similarity = np.dot(llava_embedding, landmark_embedding) / (
            np.linalg.norm(llava_embedding) * np.linalg.norm(landmark_embedding)
        )
        
        # Normalize to 0-1 range
        return float((similarity + 1) / 2)
    
    def compute_visual_similarity(
        self,
        query_clip_embedding: np.ndarray,
        landmark_clip_embedding: np.ndarray
    ) -> float:
        """
        Compute visual similarity using CLIP embeddings.
        
        Args:
            query_clip_embedding: CLIP embedding of query image
            landmark_clip_embedding: CLIP embedding of landmark image
            
        Returns:
            Similarity score (0-1)
        """
        if query_clip_embedding is None or landmark_clip_embedding is None:
            return 0.0
        
        # Cosine similarity (embeddings should already be normalized)
        similarity = np.dot(query_clip_embedding, landmark_clip_embedding)
        
        # Ensure in 0-1 range
        return float(max(0, min(1, (similarity + 1) / 2)))
    
    def recommend(
        self,
        itinerary_landmarks: List[str],
        llava_description: str,
        max_distance_km: float = 50.0,
        top_k: int = 5,
        clip_embedding: Optional[np.ndarray] = None,
        similarity_weight: float = 0.5,
        proximity_weight: float = 0.25,
        visual_weight: float = 0.2,
        popularity_weight: float = 0.05
    ) -> List[Recommendation]:
        """
        Generate top-K recommendations based on distance, text similarity, and visual similarity.
        
        Args:
            itinerary_landmarks: List of landmark names in user's itinerary
            llava_description: Feature description from LLaVA
            max_distance_km: Maximum distance from itinerary center
            top_k: Number of recommendations to return
            clip_embedding: Optional CLIP visual embedding for visual similarity
            similarity_weight: Weight for text similarity (0-1)
            proximity_weight: Weight for distance score (0-1)
            visual_weight: Weight for visual similarity (0-1)
            popularity_weight: Weight for popularity boost (0-1)
            
        Returns:
            List of Recommendation objects, sorted by final_score
        """
        # Find nearby landmarks
        nearby = self.find_nearby_landmarks(itinerary_landmarks, max_distance_km)
        
        if not nearby:
            return []
        
        # Score each landmark
        recommendations = []
        
        for landmark in nearby:
            # 1. Text similarity score (use pre-computed embedding if available)
            idx = landmark.get('_index')
            landmark_desc = landmark.get('description', landmark['name'])
            text_similarity = self.compute_similarity(
                llava_description, 
                landmark_idx=idx,
                landmark_description=landmark_desc if idx is None else None
            )
            
            # 2. Visual similarity score (if CLIP embedding provided)
            visual_similarity = 0.0
            if clip_embedding is not None and self.clip_embeddings is not None:
                # Get landmark's CLIP embedding if available
                landmark_id = landmark.get('landmark_id')
                if landmark_id in self.clip_id_to_idx:
                    emb_idx = self.clip_id_to_idx[landmark_id]
                    landmark_clip = self.clip_embeddings[emb_idx]
                    # Check if not a zero vector (means no images for this landmark)
                    if np.linalg.norm(landmark_clip) > 0:
                        visual_similarity = self.compute_visual_similarity(
                            clip_embedding, 
                            landmark_clip
                        )
            
            # 3. Proximity score (closer = better)
            proximity = 1 - (landmark['distance_to_closest'] / max_distance_km)
            proximity = max(0, proximity)  # Ensure non-negative
            
            # 4. Popularity score (based on image count)
            image_count = landmark.get('image_count', 0)
            popularity = min(image_count / 1000, 1.0)  # Normalize
            
            # Combined score with dynamic weight normalization
            # Redistribute visual_weight if either user has no image OR landmark has no visual data
            has_user_image = clip_embedding is not None
            has_landmark_visual = visual_similarity > 0  # Will be 0 if no CLIP embedding for landmark
            
            if not has_user_image or not has_landmark_visual:
                # No visual comparison possible - redistribute visual_weight proportionally
                total_other_weights = similarity_weight + proximity_weight + popularity_weight
                if total_other_weights > 0:
                    # Redistribute visual_weight proportionally to other factors
                    text_boost = visual_weight * (similarity_weight / total_other_weights)
                    proximity_boost = visual_weight * (proximity_weight / total_other_weights)
                    popularity_boost = visual_weight * (popularity_weight / total_other_weights)
                    
                    final_score = (
                        text_similarity * (similarity_weight + text_boost) +
                        proximity * (proximity_weight + proximity_boost) +
                        popularity * (popularity_weight + popularity_boost)
                    )
                else:
                    final_score = text_similarity
            else:
                # Full scoring with all factors
                final_score = (
                    text_similarity * similarity_weight +
                    visual_similarity * visual_weight +
                    proximity * proximity_weight +
                    popularity * popularity_weight
                )
            
            rec = Recommendation(
                name=landmark['name'],
                landmark_id=landmark['landmark_id'],
                latitude=landmark['latitude'],
                longitude=landmark['longitude'],
                distance_km=landmark['distance_to_closest'],
                similarity_score=text_similarity,
                final_score=final_score,
                country=landmark.get('country', 'Unknown'),
                description=landmark.get('description', ''),
                closest_itinerary_item=landmark['closest_itinerary_item']
            )
            
            recommendations.append(rec)
        
        # Sort by final score (descending)
        recommendations.sort(key=lambda x: x.final_score, reverse=True)
        
        return recommendations[:top_k]
    
    def get_available_landmarks(self) -> List[str]:
        """Get list of all available landmark names."""
        return sorted([lm['name'] for lm in self.landmarks])


# Quick test
if __name__ == '__main__':
    engine = RecommendationEngine()
    
    # Test itinerary
    itinerary = ["Eiffel Tower", "Louvre"]
    llava_desc = "gothic architecture, stained glass windows, historic cathedral, riverside"
    
    print(f"\nItinerary: {itinerary}")
    print(f"Looking for: {llava_desc}")
    print("\nTop 5 Recommendations:")
    
    recs = engine.recommend(itinerary, llava_desc, max_distance_km=20, top_k=5)
    
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec.name} (Score: {rec.final_score:.2f})")
        print(f"   Distance: {rec.distance_km:.1f}km from {rec.closest_itinerary_item}")
        print(f"   Similarity: {rec.similarity_score:.0%} | Country: {rec.country}")
        print(f"   {rec.description[:100]}...")

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
            (latitude, longitude) of center point, or (0, 0) for global search
        """
        if not itinerary_landmarks:
            return 0.0, 0.0  # Global search mode
            
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
        max_distance_km: Optional[float] = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Find landmarks within radius of itinerary center using spatial index.
        If no itinerary provided, returns all landmarks (global search).
        
        Args:
            itinerary_landmarks: List of landmark names (empty for global search)
            max_distance_km: Maximum distance from center (None for global search)
            
        Returns:
            List of nearby landmarks with distances
        """
        # Global search mode: no itinerary, return all landmarks
        if not itinerary_landmarks or max_distance_km is None:
            return [
                {
                    **landmark, 
                    'distance_km': 0.0,
                    'distance_to_closest': 0.0,
                    'closest_itinerary_item': 'Global Search',
                    '_index': idx
                }
                for idx, landmark in enumerate(self.landmarks)
            ]
        
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
        max_distance_km: Optional[float] = 50.0,
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
            itinerary_landmarks: List of landmark names in user's itinerary (empty for global)
            llava_description: Feature description from LLaVA
            max_distance_km: Maximum distance from itinerary center (None for global search)
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
            if max_distance_km is not None:
                proximity = 1 - (landmark['distance_to_closest'] / max_distance_km)
                proximity = max(0, proximity)  # Ensure non-negative
            else:
                # Global search: no proximity score
                proximity = 0.0
            
            # 4. Popularity score (based on image count)
            image_count = landmark.get('image_count', 0)
            popularity = min(image_count / 1000, 1.0)  # Normalize
            
            # Combined score with dynamic weight normalization
            is_global_search = max_distance_km is None
            has_user_image = clip_embedding is not None
            has_landmark_visual = visual_similarity > 0
            
            # Determine which weights need redistribution
            weights_to_redistribute = []
            active_weights = {}
            
            if not has_user_image or not has_landmark_visual:
                weights_to_redistribute.append(('visual', visual_weight))
            else:
                active_weights['visual'] = (visual_similarity, visual_weight)
            
            if is_global_search:
                weights_to_redistribute.append(('proximity', proximity_weight))
            else:
                active_weights['proximity'] = (proximity, proximity_weight)
            
            # Text and popularity always active
            active_weights['text'] = (text_similarity, similarity_weight)
            active_weights['popularity'] = (popularity, popularity_weight)
            
            # Redistribute inactive weights proportionally to active weights
            if weights_to_redistribute:
                total_redistribute = sum(w for _, w in weights_to_redistribute)
                total_active = sum(w for _, w in active_weights.values())
                
                if total_active > 0:
                    boost_factor = total_redistribute / total_active
                    final_score = sum(
                        score * weight * (1 + boost_factor)
                        for score, weight in active_weights.values()
                    )
                else:
                    final_score = text_similarity
            else:
                # All weights active
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
    
    
    def search_by_description(
        self,
        llava_description: str,
        clip_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search entire landmark database using LLaVA description keywords.
        No itinerary needed - pure content-based search.
        
        Args:
            llava_description: Feature description from LLaVA
            clip_embedding: Optional CLIP visual embedding
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching landmarks with scores
        """
        # Embed the query description
        query_embedding = self.embedder.encode([llava_description])[0]
        
        results = []
        for idx, landmark in enumerate(self.landmarks):
            # Text similarity using pre-computed embeddings
            if hasattr(self, 'landmark_embeddings'):
                text_sim = np.dot(query_embedding, self.landmark_embeddings[idx]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(self.landmark_embeddings[idx])
                )
                text_sim = (text_sim + 1) / 2  # Normalize to 0-1
            else:
                text_sim = 0.5
            
            # Visual similarity if CLIP available
            visual_sim = 0.0
            if clip_embedding is not None and self.clip_embeddings is not None:
                # Look up embedding index for this landmark
                lm_id = landmark['landmark_id']
                if lm_id in self.clip_id_to_idx:
                    clip_idx = self.clip_id_to_idx[lm_id]
                    lm_clip = self.clip_embeddings[clip_idx]
                    visual_sim = np.dot(clip_embedding, lm_clip)
                    visual_sim = max(0, min(1, (visual_sim + 1) / 2))
            
            # Combined score (60% text, 40% visual if available)
            if visual_sim > 0:
                final_score = 0.6 * text_sim + 0.4 * visual_sim
            else:
                final_score = text_sim
            
            # Only include if above threshold
            if final_score >= min_similarity:
                results.append({
                    'landmark_id': landmark['landmark_id'],
                    'name': landmark['name'],
                    'description': landmark.get('description', ''),
                    'country': landmark.get('country', 'Unknown'),
                    'latitude': landmark.get('latitude'),
                    'longitude': landmark.get('longitude'),
                    'text_similarity': float(text_sim),
                    'visual_similarity': float(visual_sim),
                    'final_score': float(final_score)
                })
        
        # Sort by score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results[:top_k]


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

"use client";

import React from "react";
import { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import { useRouter } from "next/navigation";
import ModernSearchSuggestions from "@/components/ModernSearchSuggestions";

const MapComponent = dynamic(() => import("@/components/MapComponent"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-zinc-900">
      <p className="text-stone-400">Loading map...</p>
    </div>
  ),
});

interface Location {
  id: string;
  name: string;
  lat: number;
  lng: number;
  image?: string;
  confidence?: number;
  day: number;
  time?: string;
  notes?: string;
}

// Constants
const POPULAR_LANDMARKS = ['tokyo tower', 'tokyo skytree', 'eiffel tower', 'louvre', 'big ben', 'london eye', 'statue of liberty', 'times square', 'colosseum', 'sagrada familia'];

// 50 distinct colors for different days (matches MapComponent)
const dayColors = [
  '#f97316', '#3b82f6', '#10b981', '#8b5cf6', '#ef4444',
  '#f59e0b', '#06b6d4', '#84cc16', '#ec4899', '#6366f1',
  '#14b8a6', '#f43f5e', '#a855f7', '#eab308', '#0ea5e9',
  '#22c55e', '#d946ef', '#fb923c', '#2563eb', '#059669',
  '#7c3aed', '#dc2626', '#ca8a04', '#0891b2', '#65a30d',
  '#db2777', '#4f46e5', '#0d9488', '#e11d48', '#9333ea',
  '#facc15', '#0284c7', '#16a34a', '#c026d3', '#fb7185',
  '#818cf8', '#14b8a6', '#f87171', '#fbbf24', '#38bdf8',
  '#4ade80', '#a78bfa', '#fb7185', '#fde047', '#22d3ee',
  '#86efac', '#c4b5fd', '#fca5a5', '#fde68a', '#7dd3fc'
];

const DESTINATION_IMAGES: { [key: string]: string } = {
  'japan': 'https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=800&q=80',
  'france': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&q=80',
  'italy': 'https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?w=800&q=80',
  'spain': 'https://images.unsplash.com/photo-1543783207-ec64e4d95325?w=800&q=80',
  'united kingdom': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800&q=80',
  'greece': 'https://images.unsplash.com/photo-1613395877344-13d4a8e0d49e?w=800&q=80',
  'thailand': 'https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?w=800&q=80',
  'australia': 'https://images.unsplash.com/photo-1523482580672-f109ba8cb9be?w=800&q=80',
  'california': 'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800&q=80',
  'hawaii': 'https://images.unsplash.com/photo-1542259009477-d625272157b7?w=800&q=80',
  'iceland': 'https://images.unsplash.com/photo-1504829857797-ddff29c27927?w=800&q=80',
  'new zealand': 'https://images.unsplash.com/photo-1507699622108-4be3abd695ad?w=800&q=80',
};

const COUNTRY_LANDMARKS: { [key: string]: string[] } = {
  'japan': ['tokyo tower', 'tokyo skytree'],
  'france': ['eiffel tower', 'louvre'],
  'united kingdom': ['big ben', 'london eye'],
  'uk': ['big ben', 'london eye'],
  'london': ['big ben', 'london eye'],
  'united states': ['statue of liberty', 'times square'],
  'usa': ['statue of liberty', 'times square'],
  'new york': ['statue of liberty', 'times square'],
  'italy': ['colosseum'],
  'rome': ['colosseum'],
  'spain': ['sagrada familia'],
  'barcelona': ['sagrada familia']
};

// Helper functions
const sortSuggestionsByRelevance = (landmarks: any[], queryLower: string) => {
  return landmarks.sort((a: any, b: any) => {
    const aLower = a.name.toLowerCase();
    const bLower = b.name.toLowerCase();
    
    // Check if name starts with query (highest priority)
    const aStarts = aLower.startsWith(queryLower);
    const bStarts = bLower.startsWith(queryLower);
    if (aStarts && !bStarts) return -1;
    if (!aStarts && bStarts) return 1;
    
    // Check if any word in the name starts with query
    const aWordStarts = aLower.split(' ').some((word: string) => word.startsWith(queryLower));
    const bWordStarts = bLower.split(' ').some((word: string) => word.startsWith(queryLower));
    if (aWordStarts && !bWordStarts) return -1;
    if (!aWordStarts && bWordStarts) return 1;
    
    // Check if it's a popular landmark
    const aPopular = POPULAR_LANDMARKS.some(p => aLower.includes(p) || p.includes(aLower));
    const bPopular = POPULAR_LANDMARKS.some(p => bLower.includes(p) || p.includes(bLower));
    if (aPopular && !bPopular) return -1;
    if (!aPopular && bPopular) return 1;
    
    return 0;
  });
};

export default function Dashboard() {
  const router = useRouter();
  const [tripName, setTripName] = useState('My Trip');
  const [destination, setDestination] = useState('');
  const [destinationImage, setDestinationImage] = useState('');
  const [destinationLat, setDestinationLat] = useState(35.6762);
  const [destinationLng, setDestinationLng] = useState(139.6503);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [isEditingDates, setIsEditingDates] = useState(false);
  const [locations, setLocations] = useState<Location[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null);
  const [uploading, setUploading] = useState(false);
  const [currentDay, setCurrentDay] = useState(1);
  const [daySubheadings, setDaySubheadings] = useState<{ [key: number]: string }>({});
  const [editingNote, setEditingNote] = useState<string | null>(null);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [searchSuggestions, setSearchSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [searchInput, setSearchInput] = useState('');
  const [insertSearchInput, setInsertSearchInput] = useState<{ [key: string]: string }>({});
  const [insertSearchSuggestions, setInsertSearchSuggestions] = useState<{ [key: string]: string[] }>({});
  const [showInsertSuggestions, setShowInsertSuggestions] = useState<{ [key: string]: boolean }>({});
  const [bottomSearchInput, setBottomSearchInput] = useState<{ [key: number]: string }>({});
  const [bottomSearchSuggestions, setBottomSearchSuggestions] = useState<{ [key: number]: string[] }>({});
  const [showBottomSuggestions, setShowBottomSuggestions] = useState<{ [key: number]: boolean }>({});
  const [destinationInput, setDestinationInput] = useState('');
  const [destinationSuggestions, setDestinationSuggestions] = useState<string[]>([]);
  const [showDestinationSuggestions, setShowDestinationSuggestions] = useState(false);
  const [isEditingDestination, setIsEditingDestination] = useState(false);
  const [showMoreInfo, setShowMoreInfo] = useState<string | null>(null);
  const [placeDetailsCache, setPlaceDetailsCache] = useState<{ [key: string]: any }>({});
  const [loadingPlaceDetails, setLoadingPlaceDetails] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [modalLocation, setModalLocation] = useState<Location | null>(null);
  const [recommendations, setRecommendations] = useState<{ name: string; lat: number; lng: number; confidence: number }[]>([]);
  const [selectedPhotoIndex, setSelectedPhotoIndex] = useState<number | null>(null);
  const [currentPhotos, setCurrentPhotos] = useState<string[]>([]);
  const [copiedItem, setCopiedItem] = useState<string | null>(null);
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareableLink, setShareableLink] = useState('');
  const [predictionModal, setPredictionModal] = useState<{
    show: boolean;
    predictionId: string;
    predictions: any[];
    imagePreview: string;
    googleVisionResult?: { landmark_name: string; confidence: number; locations: any[]; photos?: any[]; googleLocation?: any; city?: string; country?: string } | null;
  } | null>(null);
  const [loadingFallback, setLoadingFallback] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadModalDay, setUploadModalDay] = useState<number>(1);
  const [showFallbackModal, setShowFallbackModal] = useState(false);
  const [fallbackConfirmModal, setFallbackConfirmModal] = useState<{
    show: boolean;
    landmarkName: string;
    visionDescription: string;
    image: string;
    latitude: number;
    longitude: number;
    confidence: number;
  } | null>(null);
  
  // Store current prediction ID for Tier 2 fallback
  const [currentPredictionId, setCurrentPredictionId] = useState<string | null>(null);
  
  // Debounce timers
  const searchDebounceRef = useRef<NodeJS.Timeout | null>(null);
  const bottomSearchDebounceRef = useRef<{ [key: number]: NodeJS.Timeout }>({});
  const insertSearchDebounceRef = useRef<{ [key: string]: NodeJS.Timeout }>({});
  const destinationDebounceRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const dest = params.get('destination');
    const name = params.get('name');
    const start = params.get('start');
    const end = params.get('end');
    const lat = params.get('lat');
    const lng = params.get('lng');

    if (dest) setDestination(dest);
    if (name) setTripName(name);
    if (start) setStartDate(start);
    if (end) setEndDate(end);
    if (lat && lat !== 'undefined') {
      const parsedLat = parseFloat(lat);
      if (!isNaN(parsedLat)) setDestinationLat(parsedLat);
    }
    if (lng && lng !== 'undefined') {
      const parsedLng = parseFloat(lng);
      if (!isNaN(parsedLng)) setDestinationLng(parsedLng);
    }

    if (dest) {
      const destLower = dest.toLowerCase();
      if (DESTINATION_IMAGES[destLower]) {
        setDestinationImage(DESTINATION_IMAGES[destLower]);
      }
    }
  }, []);

  const fetchPlacePhotos = async (placeName: string) => {
    try {
      console.log(`[Frontend] Fetching photos for: ${placeName}`);
      const response = await fetch(
        `/api/landmarks/place-details?name=${encodeURIComponent(placeName)}`
      );
      const data = await response.json();
      console.log(`[Frontend] Photo data received:`, data);
      
      if (data.error) {
        console.error('[Frontend] API returned error:', data.error);
      }
      
      if (data.photos && data.photos.length > 0) {
        return {
          photos: data.photos,
          location: data.location,
          city: data.city || null,
          country: data.country || null
        };
      }
    } catch (error) {
      console.error('[Frontend] Error fetching place photos:', error);
    }
    return { photos: [], location: null, city: null, country: null };
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    
    console.log('Uploading file for prediction:', files[0].name);
    
    try {
      const formData = new FormData();
      formData.append('file', files[0]);
      
      // Call Lambda /predict endpoint with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
      
      const response = await fetch('https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod/predict', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error:', response.status, errorText);
        throw new Error(`Prediction failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Prediction result:', result);
      
      // Show predictions in modal for user to accept/reject
      if (result.predictions && result.predictions.length > 0) {
        // Create image preview URL
        const imagePreview = URL.createObjectURL(files[0]);
        
        // Fetch photos for each prediction
        const predictionsWithPhotos = await Promise.all(
          result.predictions.map(async (pred: any) => {
            const placeData = await fetchPlacePhotos(pred.landmark);
            return { 
              ...pred, 
              photos: placeData.photos,
              googleLocation: placeData.location,
              city: placeData.city,
              country: placeData.country
            };
          })
        );
        
        // Fetch photos for Google Vision result if present
        let googleVisionWithPhotos = result.google_vision_result;
        if (result.google_vision_result) {
          const visionPlaceData = await fetchPlacePhotos(result.google_vision_result.landmark_name);
          googleVisionWithPhotos = {
            ...result.google_vision_result,
            photos: visionPlaceData.photos,
            googleLocation: visionPlaceData.location,
            city: visionPlaceData.city,
            country: visionPlaceData.country
          };
        }
        
        setPredictionModal({
          show: true,
          predictionId: result.prediction_id,
          predictions: predictionsWithPhotos,
          imagePreview: imagePreview,
          googleVisionResult: googleVisionWithPhotos,
        });
        setUploading(false);
        setShowUploadModal(false);
      } else {
        throw new Error('No predictions returned');
      }
      
    } catch (error) {
      console.error('Error uploading image:', error);
      
      let errorMsg = 'Unknown error';
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMsg = 'Request timed out after 60 seconds. The Lambda function may be cold starting or the image is too large.';
        } else if (error.message && error.message.includes('Failed to fetch')) {
          errorMsg = 'Network error: Could not connect to the API. Check your internet connection and ensure the Lambda function is deployed.';
        } else {
          errorMsg = error.message || 'Unknown error occurred';
        }
      }
      
      alert(`Failed to identify landmark:\n${errorMsg}\n\nCheck console for details.`);
      setUploading(false);
    }
  };

  const handleAcceptPrediction = (prediction: any) => {
    if (!predictionModal) return;
    
    // Use Google Places location first, then prediction coords, then fallback
    let lat = destinationLat + Math.random() * 0.1 - 0.05;
    let lng = destinationLng + Math.random() * 0.1 - 0.05;
    
    if (prediction.googleLocation) {
      lat = prediction.googleLocation.latitude;
      lng = prediction.googleLocation.longitude;
    } else if (prediction.latitude && prediction.longitude) {
      lat = prediction.latitude;
      lng = prediction.longitude;
    }
    
    // Use the first Google Places photo if available, otherwise no image
    const landmarkImage = prediction.photos && prediction.photos.length > 0 
      ? prediction.photos[0].url 
      : undefined;
    
    const newLocation: Location = {
      id: predictionModal.predictionId,
      name: prediction.landmark,
      lat,
      lng,
      image: landmarkImage,
      confidence: prediction.confidence,
      day: currentDay,
    };
    
    setLocations([...locations, newLocation]);
    setPredictionModal(null);
  };

  const handleRejectPrediction = async () => {
    if (!predictionModal) return;
    
    // Store prediction ID for potential Tier 3 use
    setCurrentPredictionId(predictionModal.predictionId);
    
    // Close prediction modal and show purple fallback modal
    setPredictionModal(null);
    setShowFallbackModal(true);
    setLoadingFallback(true);
    
    try {
      console.log('Triggering fallback for prediction:', predictionModal.predictionId);
      
      // Call fallback endpoint with CLIP + Groq (URL-encode prediction ID)
      const encodedPredictionId = encodeURIComponent(predictionModal.predictionId);
      const response = await fetch(
        `https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod/predict/fallback/${encodedPredictionId}`,
        { method: 'POST' }
      );
      
      if (!response.ok) {
        throw new Error(`Fallback failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('Fallback result:', result);
      
      // Process top recommendation
      if (result.recommendations && result.recommendations.length > 0) {
        const topRec = result.recommendations[0];
        console.log('Top recommendation:', topRec);
        
        // Show confirmation modal with AI analysis
        // Prioritize vision_description (Groq analyzed the actual image) over topRec.description (database description)
        const descriptionSource = result.vision_description || topRec.description || 'No description available';
        // Strip markdown formatting (** **) from description
        const cleanDescription = descriptionSource
          .replace(/\*\*(.*?)\*\*/g, '$1')  // Remove **bold** markers
          .replace(/\*(.*?)\*/g, '$1');      // Remove *italic* markers
        
        // Extract landmark name from Groq vision description if available
        let landmarkName = topRec.name;
        if (result.vision_description) {
          // Try to extract landmark name from first sentence (e.g., "The landmark in the image is the Itsukushima Shrine")
          const nameMatch = result.vision_description.match(/(?:landmark in the image is|This is|famous|iconic)\s+(?:the\s+)?([^,\.]+(?:Shrine|Temple|Tower|Palace|Castle|Gate|Bridge|Monument|Building|Cathedral|Mosque|Church|Fort|Wall|Statue|Garden|Park|Falls|Mountain|Volcano|Island)[^,\.]*)/i);
          if (nameMatch) {
            landmarkName = nameMatch[1].trim();
          }
        }
        
        // Fetch photos and location for the GROQ-IDENTIFIED landmark (not CLIP match)
        // This ensures photos match what Groq identified, not what CLIP guessed
        const placeData = await fetchPlacePhotos(landmarkName);
        console.log('Place data for Groq-identified landmark', landmarkName, ':', placeData);
        
        // Use coordinates from Groq-identified place, fallback to CLIP recommendation, then fallback coords
        const latitude = placeData.location?.latitude || topRec.latitude || (destinationLat + Math.random() * 0.1 - 0.05);
        const longitude = placeData.location?.longitude || topRec.longitude || (destinationLng + Math.random() * 0.1 - 0.05);
        
        const imageUrl = placeData.photos && placeData.photos.length > 0 ? placeData.photos[0].url : '';
        
        // Close purple loading modal and show confirmation modal
        setShowFallbackModal(false);
        setLoadingFallback(false);
        
        setFallbackConfirmModal({
          show: true,
          landmarkName: landmarkName,
          visionDescription: cleanDescription,
          image: imageUrl,
          latitude,
          longitude,
          confidence: topRec.final_score || topRec.text_similarity || 0.85
        });
      } else {
        setShowFallbackModal(false);
        setLoadingFallback(false);
        alert('No recommendations found. Please try a different photo.');
      }
      
    } catch (error) {
      console.error('Error in fallback:', error);
      alert('Fallback analysis failed. Please try again.');
      setShowFallbackModal(false);
      setLoadingFallback(false);
    }
  };

  const handleConfirmFallbackLocation = () => {
    if (!fallbackConfirmModal) return;
    
    const newLocation: Location = {
      id: Date.now().toString(),
      name: fallbackConfirmModal.landmarkName,
      lat: fallbackConfirmModal.latitude,
      lng: fallbackConfirmModal.longitude,
      confidence: fallbackConfirmModal.confidence,
      day: currentDay,
      image: fallbackConfirmModal.image || undefined
    };
    
    setLocations([...locations, newLocation]);
    setFallbackConfirmModal(null);
  };
  const handleRejectFallbackLocation = () => {
    // Simply close the Tier 2 modal - no Tier 3 anymore
    setFallbackConfirmModal(null);
    setShowFallbackModal(false);
  };

  const getTotalDays = () => {
    if (!startDate || !endDate) return 1;
    const start = new Date(startDate);
    const end = new Date(endDate);
    const days = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)) + 1;
    return days > 0 ? days : 1;
  };

  const getLocationsByDay = (day: number) => {
    return locations.filter(loc => loc.day === day);
  };

  const fetchPlaceDetails = async (locationId: string, locationName: string, lat: number, lng: number) => {
    // Check cache first
    if (placeDetailsCache[locationId]) {
      return; // Already cached, no need to fetch
    }
    
    setLoadingPlaceDetails(locationId);
    try {
      const response = await fetch(`/api/place-details?name=${encodeURIComponent(locationName)}&lat=${lat}&lng=${lng}`);
      const data = await response.json();
      console.log('Place details for', locationName, ':', data);
      
      // Even if API fails, store basic info
      if (!data.rating && !data.photos?.length) {
        setPlaceDetailsCache(prev => ({ 
          ...prev, 
          [locationId]: {
            name: locationName,
            rating: null,
            totalRatings: 0,
            description: 'Details not available',
            address: `${lat}, ${lng}`,
            openingHours: '',
            website: '',
            phone: '',
            photos: [],
            reviews: [],
            priceLevel: null,
            types: [],
            isOpen: null
          }
        }));
      } else {
        setPlaceDetailsCache(prev => ({ ...prev, [locationId]: data }));
      }
    } catch (error) {
      console.error('Error fetching place details:', error);
      setPlaceDetailsCache(prev => ({ 
        ...prev, 
        [locationId]: {
          name: locationName,
          rating: null,
          totalRatings: 0,
          description: 'Failed to load details',
          address: `${lat}, ${lng}`,
          openingHours: '',
          website: '',
          phone: '',
          photos: [],
          reviews: [],
          priceLevel: null,
          types: [],
          isOpen: null
        }
      }));
    } finally {
      setLoadingPlaceDetails(null);
    }
  };

  const getLocationImage = async (locationName: string, lat?: number, lng?: number): Promise<string> => {
    // Try to fetch image from API first (Google Places photos)
    if (lat && lng) {
      try {
        const response = await fetch(`/api/place-details?name=${encodeURIComponent(locationName)}&lat=${lat}&lng=${lng}`);
        const data = await response.json();
        // Use first photo from Google Places if available
        if (data.photos && data.photos.length > 0) {
          return data.photos[0];
        }
      } catch (error) {
        console.error('Error fetching image from API:', error);
      }
    }
    
    // Fallback to hardcoded images for popular landmarks if API fails
    const landmarkImages: { [key: string]: string } = {
      'tokyo tower': 'https://images.unsplash.com/photo-1513407030348-c983a97b98d8?w=400&h=300&fit=crop',
      'tokyo': 'https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400&h=300&fit=crop',
      'eiffel tower': 'https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=400&h=300&fit=crop',
      'paris': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=400&h=300&fit=crop',
      'louvre': 'https://images.unsplash.com/photo-1499856871958-5b9627545d1a?w=400&h=300&fit=crop',
      'london': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=400&h=300&fit=crop',
      'big ben': 'https://images.unsplash.com/photo-1505761671935-60b3a7427bad?w=400&h=300&fit=crop',
      'london eye': 'https://images.unsplash.com/photo-1568849676085-51415703900f?w=400&h=300&fit=crop',
      'statue of liberty': 'https://images.unsplash.com/photo-1485738422979-f5c462d49f74?w=400&h=300&fit=crop',
      'new york': 'https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=400&h=300&fit=crop',
      'times square': 'https://images.unsplash.com/photo-1560086375-d9cb2274f2a4?w=400&h=300&fit=crop',
      'colosseum': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=400&h=300&fit=crop',
      'rome': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=400&h=300&fit=crop',
      'barcelona': 'https://images.unsplash.com/photo-1562883676-8c7feb83f09b?w=400&h=300&fit=crop',
      'sagrada familia': 'https://images.unsplash.com/photo-1523531294919-4bcd7c65e216?w=400&h=300&fit=crop',
    };
    
    const nameLower = locationName.toLowerCase();
    if (landmarkImages[nameLower]) {
      return landmarkImages[nameLower];
    }
    
    // Final fallback to generic image
    return `https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80`;
  };

  const handleAddPlace = async (day: number, placeName: string, insertAfterIndex?: number) => {
    if (!placeName.trim()) return;
    
    let lat = destinationLat + (Math.random() - 0.5) * 0.1;
    let lng = destinationLng + (Math.random() - 0.5) * 0.1;
    let imageUrl = '';
    
    try {
      // Try Google Places search first for better accuracy
      const response = await fetch('/api/landmarks/google-search?q=' + encodeURIComponent(placeName));
      const data = await response.json();
      console.log('Google search result for', placeName, ':', data.landmarks?.[0]);
      if (data.landmarks && data.landmarks.length !== 0 && data.landmarks[0].latitude && data.landmarks[0].longitude) {
        lat = data.landmarks[0].latitude;
        lng = data.landmarks[0].longitude;
        // Use photo from search if available
        imageUrl = data.landmarks[0].photo || '';
        console.log('Using photo from search:', imageUrl);
      }
    } catch (error) {
      console.error('Error fetching coordinates:', error);
    }
    
    // Get image if not already set
    if (!imageUrl) {
      console.log('No photo from search, fetching place details for', placeName);
      imageUrl = await getLocationImage(placeName, lat, lng);
      console.log('Got image from place details:', imageUrl);
    }
    
    const newLocation: Location = {
      id: Date.now().toString(),
      name: placeName,
      lat: lat,
      lng: lng,
      image: imageUrl,
      day: day,
      notes: '',
    };
    
    if (insertAfterIndex !== undefined) {
      const dayLocs = locations.filter(loc => loc.day === day);
      const otherDayLocs = locations.filter(loc => loc.day !== day);
      dayLocs.splice(insertAfterIndex + 1, 0, newLocation);
      setLocations([...otherDayLocs, ...dayLocs]);
    } else {
      setLocations([...locations, newLocation]);
    }
  };

  const handleAddFromMap = async (placeName: string, lat: number, lng: number) => {
    // Get image using place details which has better photo quality
    const imageUrl = await getLocationImage(placeName, lat, lng);
    
    // Find the last day with locations, or use day 1
    const existingDays = [...new Set(locations.map(loc => loc.day))].sort((a, b) => b - a);
    const targetDay = existingDays.length > 0 ? existingDays[0] : 1;
    
    const newLocation: Location = {
      id: Date.now().toString(),
      name: placeName,
      lat: lat,
      lng: lng,
      image: imageUrl,
      day: targetDay,
      notes: '',
    };
    setLocations([...locations, newLocation]);
  };

  const handleDeleteLocation = (locationId: string) => {
    setLocations(locations.filter(loc => loc.id !== locationId));
    if (selectedLocation?.id === locationId) {
      setSelectedLocation(null);
    }
  };

  const handleDragStart = (e: React.DragEvent, locationId: string) => {
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', locationId);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent, targetLocationId: string, day: number) => {
    e.preventDefault();
    const draggedId = e.dataTransfer.getData('text/plain');
    
    if (draggedId === targetLocationId) return;
    
    const draggedLocation = locations.find(loc => loc.id === draggedId);
    if (!draggedLocation) return;
    
    const dayLocations = locations.filter(loc => loc.day === day);
    const otherLocations = locations.filter(loc => loc.day !== day);
    
    const draggedIndex = dayLocations.findIndex(loc => loc.id === draggedId);
    const targetIndex = dayLocations.findIndex(loc => loc.id === targetLocationId);
    
    if (draggedIndex !== -1) {
      dayLocations.splice(draggedIndex, 1);
    }
    
    const updatedDragged = { ...draggedLocation, day: day };
    
    if (targetIndex !== -1) {
      dayLocations.splice(targetIndex, 0, updatedDragged);
    } else {
      dayLocations.push(updatedDragged);
    }
    
    setLocations([...otherLocations, ...dayLocations]);
  };

  const handleFindLocations = async (day: number) => {
    // Check if destination is set
    if (!destination || !destination.trim()) {
      // Show minimalist error popup
      const errorDiv = document.createElement('div');
      errorDiv.className = 'fixed top-24 left-1/2 -translate-x-1/2 bg-zinc-900 border border-orange-500/30 rounded-xl px-6 py-4 shadow-2xl z-[10000] animate-fadeIn';
      errorDiv.innerHTML = `
        <div class="flex items-center gap-3">
          <svg class="w-5 h-5 text-orange-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p class="text-stone-300 text-sm">Please set a destination first</p>
        </div>
      `;
      document.body.appendChild(errorDiv);
      setTimeout(() => {
        errorDiv.style.opacity = '0';
        errorDiv.style.transition = 'opacity 300ms';
        setTimeout(() => errorDiv.remove(), 300);
      }, 3000);
      return;
    }
    
    console.log('🔍 Discover clicked! Day:', day);
    setLoadingRecommendations(true);
    setCurrentDay(day); // Set the day for adding recommendations
    
    try {
      // Get auth token from localStorage
      const token = localStorage.getItem('auth_token');
      
      console.log('📍 Destination coordinates:', { destinationLat, destinationLng });
      console.log('Calling backend with:', {
        itinerary_landmarks: locations.map(loc => loc.name),
        destination,
        token: token ? 'present' : 'missing'
      });
      
      // Try the real backend /recommend endpoint first
      try {
        const response = await fetch('https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod/recommend', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token || ''}`,
          },
          body: JSON.stringify({
            itinerary_landmarks: locations.map(loc => loc.name),
            llava_description: `Popular tourist attractions and landmarks near ${destination}`,
            clip_embedding: null,
            max_distance_km: 50.0,
            top_k: 5
          }),
        });

        console.log('Response status:', response.status);
        
        if (response.ok) {
          const data = await response.json();
          console.log('Backend response:', data);
          
          // Transform backend response to match our UI format
          const transformedRecs = data.recommendations.map((rec: any) => ({
            name: rec.name,
            lat: rec.latitude,
            lng: rec.longitude,
            confidence: rec.final_score || rec.similarity_score || 0.85
          }));
          
          console.log('Transformed recommendations:', transformedRecs);
          setRecommendations(transformedRecs);
          setLoadingRecommendations(false);
          return;
        } else {
          const errorText = await response.text();
          console.warn('Backend error, falling back to Google Places:', response.status, errorText);
        }
      } catch (backendError) {
        console.warn('Backend request failed, falling back to Google Places:', backendError);
      }
      
      // Fallback: Use our API route which calls Google Places API
      console.log('Using Google Places API fallback via /api/recommendations');
      console.log('Sending to fallback API:', { destinationLat, destinationLng, existingLocations: locations.length });
      
      const placesResponse = await fetch('/api/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          destinationLat,
          destinationLng,
          existingLocations: locations
        })
      });

      console.log('Fallback API response status:', placesResponse.status);

      if (placesResponse.ok) {
        const placesData = await placesResponse.json();
        console.log('Fallback recommendations:', placesData.recommendations);
        setRecommendations(placesData.recommendations);
      } else {
        const errorText = await placesResponse.text();
        console.error('Fallback API error:', placesResponse.status, errorText);
      }
      
      setLoadingRecommendations(false);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setLoadingRecommendations(false);
    }
  };

  const handleUpdateNote = (locationId: string, note: string) => {
    setLocations(locations.map(loc => 
      loc.id === locationId ? { ...loc, notes: note } : loc
    ));
  };

  const handleAddRecommendationToItinerary = async (name: string, lat: number, lng: number) => {
    const image = await getLocationImage(name);
    const newLocation: Location = {
      id: Date.now().toString(),
      name,
      lat,
      lng,
      image,
      day: currentDay,
    };
    setLocations([...locations, newLocation]);
  };

  const handleClearRecommendations = () => {
    setRecommendations([]);
  };

  // Share functionality
  const generateShareableId = () => {
    return 'trip-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
  };

  const handleShare = () => {
    // Generate unique ID for this itinerary
    const shareId = generateShareableId();
    
    // Prepare itinerary data
    const itineraryData = {
      id: shareId,
      tripName,
      destination,
      destinationImage,
      destinationLat,
      destinationLng,
      startDate,
      endDate,
      locations,
      daySubheadings,
      timestamp: new Date().toISOString()
    };
    
    // Save to localStorage
    localStorage.setItem(`itinerary-${shareId}`, JSON.stringify(itineraryData));
    
    // Generate shareable link
    const baseUrl = window.location.origin;
    const shareLink = `${baseUrl}/itinerary/${shareId}`;
    setShareableLink(shareLink);
    setShowShareModal(true);
  };

  const handleCopyShareLink = async () => {
    try {
      await navigator.clipboard.writeText(shareableLink);
      setCopiedItem('shareLink');
      setTimeout(() => setCopiedItem(null), 1000);
    } catch (error) {
      console.error('Failed to copy link:', error);
    }
  };

  const handleSearchChange = (value: string) => {
    setSearchInput(value);
    
    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
    }
    
    if (value.trim().length > 1) {
      searchDebounceRef.current = setTimeout(async () => {
        try {
          const response = await fetch('/api/landmarks/google-search?q=' + encodeURIComponent(value));
          const data = await response.json();
          
          // API now handles all intelligent sorting, filtering, and scoring
          const suggestions = (data.landmarks || [])
            .slice(0, 8) // Top 8 results
            .map((l: any) => l.name);
          
          setSearchSuggestions(suggestions);
          setShowSuggestions(suggestions.length > 0);
        } catch (error) {
          console.error('Error searching landmarks:', error);
          setShowSuggestions(false);
        }
      }, 300);
    } else {
      setShowSuggestions(false);
    }
  };

  const handleBottomSearchChange = (day: number, value: string) => {
    setBottomSearchInput({ ...bottomSearchInput, [day]: value });
    
    if (bottomSearchDebounceRef.current[day]) {
      clearTimeout(bottomSearchDebounceRef.current[day]);
    }
    
    if (value.trim().length > 1) {
      bottomSearchDebounceRef.current[day] = setTimeout(async () => {
        try {
          const response = await fetch('/api/landmarks/google-search?q=' + encodeURIComponent(value));
          const data = await response.json();
          
          // API handles all intelligent sorting and filtering
          const suggestions = (data.landmarks || [])
            .slice(0, 8)
            .map((l: any) => l.name);
          
          setBottomSearchSuggestions({ ...bottomSearchSuggestions, [day]: suggestions });
          setShowBottomSuggestions({ ...showBottomSuggestions, [day]: suggestions.length > 0 });
        } catch (error) {
          console.error('Error searching landmarks:', error);
          setShowBottomSuggestions({ ...showBottomSuggestions, [day]: false });
        }
      }, 300);
    } else {
      setShowBottomSuggestions({ ...showBottomSuggestions, [day]: false });
    }
  };

  const handleInsertSearchChange = (key: string, value: string) => {
    setInsertSearchInput({ ...insertSearchInput, [key]: value });
    
    // Clear previous debounce for this insert position
    if (insertSearchDebounceRef.current[key]) {
      clearTimeout(insertSearchDebounceRef.current[key]);
    }
    
    if (value.trim().length > 1) {
      insertSearchDebounceRef.current[key] = setTimeout(async () => {
        try {
          const response = await fetch('/api/landmarks/google-search?q=' + encodeURIComponent(value));
          const data = await response.json();
          
          // API handles all intelligent sorting and filtering
          const suggestions = (data.landmarks || [])
            .slice(0, 8)
            .map((l: any) => l.name);
          
          setInsertSearchSuggestions({ ...insertSearchSuggestions, [key]: suggestions });
          setShowInsertSuggestions({ ...showInsertSuggestions, [key]: suggestions.length > 0 });
        } catch (error) {
          console.error('Error searching landmarks:', error);
          setShowInsertSuggestions({ ...showInsertSuggestions, [key]: false });
        }
      }, 300);
    } else {
      setShowInsertSuggestions({ ...showInsertSuggestions, [key]: false });
    }
  };

  const handleDestinationSearchChange = (value: string) => {
    setDestinationInput(value);
    
    if (destinationDebounceRef.current) {
      clearTimeout(destinationDebounceRef.current);
    }
    
    if (value.trim().length > 1) {
      destinationDebounceRef.current = setTimeout(async () => {
        try {
          const response = await fetch('/api/landmarks/google-search?q=' + encodeURIComponent(value));
          const data = await response.json();
          // Filter to only show large locations (countries, states, regions) - NO cities or smaller
          const suggestions = data.landmarks
            .filter((l: any) => {
              const types = (l.types || []).map((t: string) => t.toLowerCase());
              const name = (l.name || '').toLowerCase();
              
              // For Google Places API, check types array
              // Country-level results
              const isCountry = types.includes('country') || types.includes('administrative_area_level_1');
              
              // State/region level
              const isRegion = types.includes('administrative_area_level_1') || 
                              types.includes('administrative_area_level_2');
              
              // Exclude cities and smaller
              const isCity = types.includes('locality') || 
                            types.includes('sublocality') ||
                            types.includes('neighborhood') ||
                            types.includes('premise');
              
              // For simple queries like "japan", "france", allow all results
              if (value.trim().length < 8 && !isCity) {
                return true;
              }
              
              return (isCountry || isRegion) && !isCity;
            })
            .map((l: any) => l.name);
          setDestinationSuggestions(suggestions.slice(0, 5));
          setShowDestinationSuggestions(suggestions.length !== 0);
        } catch (error) {
          console.error('Error searching destinations:', error);
          setShowDestinationSuggestions(false);
        }
      }, 300);
    } else {
      setShowDestinationSuggestions(false);
    }
  };

  const handleSelectDestination = async (dest: string) => {
    setDestination(dest);
    setDestinationInput('');
    setShowDestinationSuggestions(false);
    setIsEditingDestination(false);
    
    // Fetch coordinates for the destination
    try {
      const response = await fetch('/api/landmarks/google-search?q=' + encodeURIComponent(dest));
      const data = await response.json();
      if (data.landmarks && data.landmarks.length > 0) {
        const location = data.landmarks[0];
        setDestinationLat(location.latitude);
        setDestinationLng(location.longitude);
      }
    } catch (error) {
      console.error('Error fetching destination coordinates:', error);
    }
    
    // Update destination image
    const destLower = dest.toLowerCase();
    if (DESTINATION_IMAGES[destLower]) {
      setDestinationImage(DESTINATION_IMAGES[destLower]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-900 via-stone-900 to-zinc-900 flex flex-col">
      <div className="relative z-[2000] bg-zinc-900 bg-opacity-70 backdrop-blur-2xl border-b border-zinc-800 border-opacity-50 px-8 py-4 shadow-2xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button onClick={() => router.push('/')} className="text-stone-400 hover:text-stone-300 transition-colors">
              <Image
                src="/images/logo.png"
                alt="TripSaver"
                width={32}
                height={32}
                quality={100}
              />
            </button>
            <div className="h-5 w-px bg-stone-700 bg-opacity-30"></div>
            <input
              type="text"
              value={tripName}
              onChange={(e) => setTripName(e.target.value)}
              placeholder="Trip Name"
              spellCheck={false}
              className="text-lg font-semibold text-white bg-transparent border-none outline-none focus:text-orange-400 transition-colors placeholder-stone-600"
            />
            <div className="h-5 w-px bg-stone-700 bg-opacity-30"></div>
            {!isEditingDestination ? (
              <button
                onClick={() => {
                  setIsEditingDestination(true);
                  setDestinationInput(destination);
                }}
                className="text-sm text-stone-400 hover:text-orange-400 transition-colors"
              >
                {destination || 'Add destination'}
              </button>
            ) : (
              <div className="relative">
                <input
                  type="text"
                  value={destinationInput}
                  onChange={(e) => handleDestinationSearchChange(e.target.value)}
                  onFocus={() => {
                    if (destinationInput.trim().length > 1) {
                      handleDestinationSearchChange(destinationInput);
                    }
                  }}
                  onBlur={() => {
                    setTimeout(() => {
                      setIsEditingDestination(false);
                      setShowDestinationSuggestions(false);
                    }, 200);
                  }}
                  placeholder="Type destination..."
                  autoFocus
                  spellCheck={false}
                  className="text-sm text-white bg-zinc-800 border border-zinc-700 rounded px-2 py-1 focus:outline-none focus:border-orange-400 min-w-[150px] placeholder-white placeholder-opacity-60"
                />
                {showDestinationSuggestions && destinationSuggestions.length > 0 && (
                  <div className="absolute top-full left-0 mt-1 z-[10001]">
                    <ModernSearchSuggestions
                      suggestions={destinationSuggestions}
                      onSelect={handleSelectDestination}
                    />
                  </div>
                )}
              </div>
            )}
            <div className="h-5 w-px bg-stone-700 bg-opacity-30"></div>
            {isEditingDates ? (
              <div className="flex items-center gap-2">
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="text-xs bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-white focus:outline-none focus:border-orange-400 [color-scheme:dark]"
                />
                <span className="text-stone-500 text-xs">-</span>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="text-xs bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-white focus:outline-none focus:border-orange-400 [color-scheme:dark]"
                />
                <button
                  onClick={() => setIsEditingDates(false)}
                  className="text-xs text-orange-400 hover:text-orange-300 ml-1"
                >
                  ✓
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsEditingDates(true)}
                className="text-sm text-white hover:text-orange-400 transition-colors"
              >
                {startDate && endDate ? `${startDate} - ${endDate}` : 'Add dates'}
              </button>
            )}
          </div>
          <button 
            onClick={handleShare}
            className="text-stone-400 hover:text-stone-300 hover:text-orange-400 px-3 py-1.5 transition-colors text-sm"
          >
            Share
          </button>
        </div>
      </div>

      <div className="flex-1 relative">
        <div className="absolute inset-0">
          <MapComponent 
            landmarks={locations} 
            selectedLandmark={selectedLocation} 
            onAddToItinerary={handleAddRecommendationToItinerary}
            recommendations={recommendations}
            onClearRecommendations={handleClearRecommendations}
          />
        </div>

        <div 
          className={`absolute top-5 bottom-5 backdrop-blur-3xl border border-zinc-800 border-opacity-50 flex flex-col shadow-[0_8px_32px_0_rgba(0,0,0,0.6)] rounded-3xl z-[1000] transition-all duration-500 ease-in-out ${sidebarCollapsed ? 'w-0 border-0 bg-transparent left-2.5' : 'w-96 min-w-[384px] bg-zinc-900 bg-opacity-95 left-5'}`}
          style={{overflow: 'visible'}}
        >
          {/* Minimize Button */}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="absolute -right-4 top-1/2 -translate-y-1/2 w-8 h-16 bg-zinc-900 bg-opacity-95 backdrop-blur-xl border border-zinc-800 border-opacity-50 rounded-r-2xl flex items-center justify-center hover:bg-zinc-800 transition-all duration-300 shadow-lg group z-[1001]"
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <svg 
              className={`w-4 h-4 text-stone-400 group-hover:text-orange-400 transition-all duration-500 ${sidebarCollapsed ? 'rotate-180' : ''}`} 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" />
            </svg>
          </button>

          <div className={`transition-opacity duration-300 flex flex-col h-full ${sidebarCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
          {(destination || tripName) && (
            <div className="relative h-48 border-b border-zinc-800 border-opacity-50 overflow-hidden rounded-t-3xl">
              {destinationImage ? (
                <div className="absolute inset-0 w-full h-full">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={destinationImage} 
                    alt={destination || 'Trip destination'}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent"></div>
                </div>
              ) : (
                <div className="absolute inset-0 bg-gradient-to-br from-orange-900/20 to-purple-900/20"></div>
              )}
              <div className="absolute bottom-4 left-5 right-5">
                <h1 className="text-2xl font-bold text-white mb-1 drop-shadow-2xl tracking-tight">{tripName}</h1>
                {destination && (
                  <p className="text-sm text-white text-opacity-90 drop-shadow-lg font-medium">{destination}</p>
                )}
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {Array.from({ length: getTotalDays() }, (_, i) => i + 1).map((day) => {
              const dayLocations = getLocationsByDay(day);
              const isExpanded = currentDay === day;
              
              return (
                <div key={day} className="mb-4 mx-4 first:mt-4 relative">
                  <div className="bg-zinc-800 bg-opacity-60 backdrop-blur-2xl rounded-3xl border border-zinc-700 border-opacity-40 overflow-hidden shadow-[0_8px_30px_rgb(0,0,0,0.3)]">
                    <div className="w-full flex items-center justify-between px-5 py-4 hover:bg-zinc-700 hover:bg-opacity-20 transition-all duration-200">
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <div 
                          className="w-9 h-9 rounded-2xl flex items-center justify-center text-white text-sm font-bold shadow-lg flex-shrink-0"
                          style={{ backgroundColor: dayColors[day % dayColors.length] }}
                        >
                          {day}
                        </div>
                        <div className="flex-1 min-w-0 text-left">
                          <div>
                            <span className="text-white text-base font-medium">Day {day}</span>
                          </div>
                          {isExpanded ? (
                            <input
                              type="text"
                              value={daySubheadings[day] || ''}
                              onChange={(e) => setDaySubheadings({ ...daySubheadings, [day]: e.target.value })}
                              placeholder="Add description..."
                              spellCheck={false}
                              autoFocus={false}
                              className="block w-full mt-1 px-0 py-0.5 bg-transparent border-none text-white text-xs placeholder-white placeholder-opacity-60 focus:outline-none focus:text-white transition-all"
                            />
                          ) : (
                            daySubheadings[day] && (
                              <p className="text-white text-opacity-60 text-xs mt-1 truncate">{daySubheadings[day]}</p>
                            )
                          )}
                        </div>
                      </div>
                      <button
                        onClick={() => setCurrentDay(isExpanded ? 0 : day)}
                        className="text-stone-500 transition-transform inline-block flex-shrink-0 ml-2 hover:text-stone-400 p-2 -m-2"
                      >
                        <span className={isExpanded ? 'rotate-180 inline-block' : 'inline-block'}>v</span>
                      </button>
                    </div>

                    {isExpanded && (
                      <div className="px-5 py-4">

                        <div className="relative mb-3">
                          <svg className="absolute left-4 top-1/2 transform -translate-y-1/2 w-4 h-4 text-white text-opacity-60 z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                          </svg>
                          <input
                            type="text"
                            value={searchInput}
                            placeholder="Add a place..."
                            spellCheck={false}
                            autoComplete="off"
                            onChange={(e) => handleSearchChange(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && searchInput.trim()) {
                                handleAddPlace(day, searchInput);
                                setSearchInput('');
                                setShowSuggestions(false);
                              }
                            }}
                            onFocus={async () => {
                              if (searchInput.trim().length > 1) {
                                try {
                                  const countryParam = destination ? `&country=${encodeURIComponent(destination)}` : '';
                                  const response = await fetch('/api/landmarks/google-search?q=' + encodeURIComponent(searchInput));
                                  const data = await response.json();
                                  
                                  const queryLower = searchInput.toLowerCase();
                                  
                                  // Only inject popular landmarks if no destination filter (to avoid showing wrong country landmarks)
                                  if (!destination) {
                                    // Inject popular landmarks that match the query but might not be in API results
                                    const matchingPopular = POPULAR_LANDMARKS.filter(landmark => {
                                      const words = landmark.split(' ');
                                      return words.some(word => word.startsWith(queryLower)) || landmark.includes(queryLower);
                                    });
                                    
                                    // Add matching popular landmarks to results if not already present
                                    const existingNames = new Set(data.landmarks.map((l: any) => l.name.toLowerCase()));
                                    matchingPopular.forEach(landmark => {
                                      if (!existingNames.has(landmark)) {
                                        data.landmarks.unshift({ name: landmark.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '), latitude: 0, longitude: 0 });
                                      }
                                    });
                                  }
                                  
                                  // Filter out results that match popular landmarks from other countries when destination is set
                                  let filteredLandmarks = data.landmarks;
                                  if (destination) {
                                    const destLower = destination.toLowerCase();
                                    const allowedLandmarks = COUNTRY_LANDMARKS[destLower] || [];
                                    filteredLandmarks = data.landmarks.filter((l: any) => {
                                      const nameLower = l.name.toLowerCase();
                                      // Keep if not a popular landmark, or if it matches the destination country
                                      const isPopularLandmark = POPULAR_LANDMARKS.some(p => nameLower.includes(p) || p.includes(nameLower));
                                      if (!isPopularLandmark) return true;
                                      return allowedLandmarks.some(allowed => nameLower.includes(allowed) || allowed.includes(nameLower));
                                    });
                                  }
                                  
                                  const suggestions = sortSuggestionsByRelevance(filteredLandmarks, queryLower).map((l: any) => l.name);
                                  setSearchSuggestions(suggestions.slice(0, 3));
                                  setShowSuggestions(suggestions.length !== 0);
                                } catch (error) {
                                  console.error('Error searching landmarks:', error);
                                }
                              }
                            }}
                            className="w-full pl-10 pr-4 py-3 bg-zinc-800 bg-opacity-50 border border-zinc-700 border-opacity-40 rounded-2xl text-white text-sm placeholder-white placeholder-opacity-60 focus:outline-none focus:border-orange-400 focus:bg-zinc-800 transition-all" 
                          />
                          {showSuggestions && searchSuggestions.length > 0 ? (
                            <div className="absolute top-full left-0 right-0 mt-1 z-[10000]">
                              <ModernSearchSuggestions
                                suggestions={searchSuggestions}
                                onSelect={(suggestion) => {
                                  handleAddPlace(day, suggestion);
                                  setSearchInput('');
                                  setShowSuggestions(false);
                                }}
                              />
                            </div>
                          ) : null}
                        </div>

                        <div className="grid grid-cols-2 gap-2 mb-3">
                          <button
                            onClick={() => handleFindLocations(day)}
                            disabled={loadingRecommendations}
                            className="bg-zinc-800 bg-opacity-40 backdrop-blur-sm border border-orange-500 border-opacity-50 text-stone-300 px-3 py-2 rounded-xl text-sm font-medium flex items-center justify-center gap-2 hover:bg-orange-600 hover:text-white hover:border-opacity-100 active:scale-95 active:bg-orange-500 active:animate-pulse transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                            Discover
                          </button>

                          <button
                            onClick={() => {
                              setCurrentDay(day);
                              setUploadModalDay(day);
                              setShowUploadModal(true);
                            }}
                            className="bg-zinc-800 bg-opacity-40 backdrop-blur-sm border border-orange-500 border-opacity-50 text-stone-300 px-3 py-2 rounded-xl text-sm font-medium cursor-pointer flex items-center justify-center gap-2 hover:bg-orange-600 hover:text-white hover:border-opacity-100 active:scale-95 active:bg-orange-500 active:animate-pulse transition-all"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                              <circle cx="8" cy="9" r="1" fill="currentColor" />
                              <circle cx="12" cy="9" r="1" fill="currentColor" />
                              <circle cx="16" cy="9" r="1" fill="currentColor" />
                            </svg>
                            Detect
                          </button>
                        </div>

                        {dayLocations.length === 0 ? (
                          <div className="py-6 text-center">
                            <p className="text-white text-opacity-60 text-sm">No locations yet</p>
                          </div>
                        ) : (
                          <div className="space-y-2">
                            {dayLocations.map((location, index) => (
                              <React.Fragment key={location.id}>
                                <div
                                  draggable
                                  onDragStart={(e) => handleDragStart(e, location.id)}
                                  onDragOver={handleDragOver}
                                  onDrop={(e) => handleDrop(e, location.id, day)}
                                  className={'rounded-3xl overflow-visible transition-all duration-300 cursor-grab active:cursor-grabbing active:border-2 active:border-orange-500 active:shadow-[0_0_20px_rgba(249,115,22,0.6)] active:animate-pulse ' + (
                                    selectedLocation?.id === location.id
                                      ? 'bg-orange-500 bg-opacity-20 border-2 border-orange-400 shadow-[0_8px_30px_rgb(249,115,22,0.4)] scale-[1.02]'
                                      : 'bg-zinc-800 bg-opacity-60 border border-zinc-700 border-opacity-40 hover:bg-opacity-70 hover:shadow-[0_8px_30px_rgb(0,0,0,0.4)] hover:scale-[1.01]'
                                  ) + (showMoreInfo === location.id ? ' min-h-[600px]' : '')}
                                >
                                  <div
                                    className="relative h-40 w-full cursor-pointer overflow-hidden rounded-t-3xl"
                                    onClick={() => setSelectedLocation(location)}
                                  >
                                    {location.image ? (
                                      <img
                                        src={location.image}
                                        alt={location.name}
                                        className="absolute inset-0 w-full h-full object-cover"
                                      />
                                    ) : (
                                      <div className="absolute inset-0 bg-gradient-to-br from-zinc-800 to-zinc-900 flex items-center justify-center">
                                        <svg className="w-16 h-16 text-stone-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                        </svg>
                                      </div>
                                    )}
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent"></div>
                                    <div 
                                      className="absolute top-3 left-3 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-xl border-2 border-white"
                                      style={{ backgroundColor: dayColors[day % dayColors.length] }}
                                    >
                                      {index + 1}
                                    </div>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleDeleteLocation(location.id);
                                      }}
                                      className="absolute top-3 right-3 w-8 h-8 bg-black bg-opacity-60 hover:bg-opacity-80 backdrop-blur-xl rounded-full flex items-center justify-center transition-all border border-white border-opacity-20 hover:scale-110"
                                    >
                                      <span className="text-stone-400 hover:text-red-400 transition-colors">X</span>
                                    </button>
                                    <h3 className="absolute bottom-2 left-2 right-2 text-white text-sm font-semibold truncate drop-shadow-lg">
                                      {location.name}
                                    </h3>
                                  </div>

                                  <div className="px-3 py-2">
                                    {editingNote === location.id ? (
                                      <textarea
                                        autoFocus
                                        value={location.notes || ''}
                                        onChange={(e) => handleUpdateNote(location.id, e.target.value)}
                                        onBlur={() => setEditingNote(null)}
                                        placeholder="Add notes..."
                                        spellCheck={false}
                                        className="w-full px-2 py-1.5 bg-zinc-900 bg-opacity-50 border border-stone-700 border-opacity-30 rounded text-white text-sm placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-50 resize-none"
                                        rows={2}
                                      />
                                    ) : (
                                      <button
                                        onClick={() => setEditingNote(location.id)}
                                        className={'w-full text-left px-2 py-1 text-sm transition-colors rounded ' + (
                                          location.notes
                                            ? 'text-white text-opacity-80'
                                            : 'text-stone-500 hover:text-stone-400 hover:bg-zinc-800 hover:bg-opacity-30'
                                        )}
                                      >
                                        {location.notes || '+ Add note'}
                                      </button>
                                    )}
                                    
                                    <div className="flex gap-2 mt-2">
                                      <button
                                        onClick={() => {
                                          if (showMoreInfo === location.id) {
                                            setShowMoreInfo(null);
                                          } else {
                                            setShowMoreInfo(location.id);
                                            fetchPlaceDetails(location.id, location.name, location.lat, location.lng);
                                          }
                                        }}
                                        className="flex-1 px-3 py-2 bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white text-sm font-medium rounded-xl transition-all hover:scale-105 hover:shadow-lg flex items-center justify-center gap-2"
                                      >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        {showMoreInfo === location.id ? 'Hide Info' : 'More Info'}
                                        <svg 
                                          className={`w-4 h-4 transition-transform duration-300 ${showMoreInfo === location.id ? 'rotate-180' : ''}`} 
                                          fill="none" 
                                          stroke="currentColor" 
                                          viewBox="0 0 24 24"
                                        >
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                        </svg>
                                      </button>
                                      <button
                                        onClick={() => {
                                          setModalLocation(location);
                                          fetchPlaceDetails(location.id, location.name, location.lat, location.lng);
                                        }}
                                        className="px-3 py-2 bg-zinc-700 bg-opacity-60 hover:bg-opacity-80 text-white text-sm font-medium rounded-xl transition-all hover:scale-105 hover:shadow-lg flex items-center justify-center"
                                        title="Maximize"
                                      >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                                        </svg>
                                      </button>
                                    </div>
                                    
                                    {/* Expandable Details Section */}
                                    {showMoreInfo === location.id && (
                                      <div className="mt-3 overflow-y-auto max-h-[450px] animate-slideDown">
                                        {loadingPlaceDetails === location.id ? (
                                          <div className="py-6 flex items-center justify-center">
                                            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-orange-500"></div>
                                          </div>
                                        ) : placeDetailsCache[location.id] ? (
                                          <div className="bg-zinc-900 bg-opacity-50 rounded-2xl p-4 space-y-3 border border-zinc-700 border-opacity-30">
                                            {placeDetailsCache[location.id].rating && (
                                              <div className="flex items-center gap-2">
                                                <div className="flex items-center gap-1">
                                                  {[...Array(5)].map((_, i) => (
                                                    <svg
                                                      key={i}
                                                      className={`w-4 h-4 ${i < Math.floor(placeDetailsCache[location.id].rating) ? 'text-yellow-400' : 'text-gray-600'}`}
                                                      fill="currentColor"
                                                      viewBox="0 0 20 20"
                                                    >
                                                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                                    </svg>
                                                  ))}
                                                </div>
                                                <span className="text-white font-semibold text-sm">{placeDetailsCache[location.id].rating.toFixed(1)}</span>
                                                {placeDetailsCache[location.id].totalRatings && (
                                                  <span className="text-stone-400 text-xs">({placeDetailsCache[location.id].totalRatings.toLocaleString()} reviews)</span>
                                                )}
                                              </div>
                                            )}
                                            
                                            {placeDetailsCache[location.id].photos && placeDetailsCache[location.id].photos.length > 0 && (
                                              <div className="space-y-2">
                                                <h4 className="text-white font-semibold text-xs flex items-center gap-1.5">
                                                  <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                                  </svg>
                                                  Photos
                                                </h4>
                                                <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-orange-500 scrollbar-track-transparent">
                                                  {placeDetailsCache[location.id].photos.map((photoUrl: string, idx: number) => (
                                                    <div key={idx} className="flex-shrink-0 w-28 h-20 rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700 border-opacity-30">
                                                      <Image
                                                        src={photoUrl}
                                                        alt={`${location.name} photo ${idx + 1}`}
                                                        width={112}
                                                        height={80}
                                                        className="w-full h-full object-cover hover:scale-110 transition-transform duration-300"
                                                      />
                                                    </div>
                                                  ))}
                                                </div>
                                              </div>
                                            )}
                                            
                                            {placeDetailsCache[location.id].description && (
                                              <div>
                                                <p className="text-stone-300 text-xs leading-relaxed">{placeDetailsCache[location.id].description}</p>
                                              </div>
                                            )}
                                            
                                            {placeDetailsCache[location.id].openingHours && (
                                              <div className="flex items-start gap-2">
                                                <svg className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                </svg>
                                                <p className="text-stone-300 text-xs">{placeDetailsCache[location.id].openingHours}</p>
                                              </div>
                                            )}
                                            
                                            {placeDetailsCache[location.id].website && (
                                              <a
                                                href={placeDetailsCache[location.id].website}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="flex items-center gap-2 text-orange-400 hover:text-orange-300 text-xs transition-colors"
                                              >
                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                                                </svg>
                                                Visit Website
                                              </a>
                                            )}
                                            
                                            {placeDetailsCache[location.id].phone && (
                                              <div className="flex items-center gap-2">
                                                <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                                                </svg>
                                                <p className="text-stone-300 text-xs">{placeDetailsCache[location.id].phone}</p>
                                              </div>
                                            )}
                                            
                                            {placeDetailsCache[location.id].address && (
                                              <div className="flex items-start gap-2">
                                                <svg className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                                </svg>
                                                <p className="text-stone-300 text-xs">{placeDetailsCache[location.id].address}</p>
                                              </div>
                                            )}
                                            
                                            <div className="flex flex-wrap gap-2">
                                              {placeDetailsCache[location.id].isOpen !== undefined && (
                                                <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-medium ${
                                                  placeDetailsCache[location.id].isOpen 
                                                    ? 'bg-green-500 bg-opacity-20 text-white border border-green-500 border-opacity-30' 
                                                    : 'bg-red-500 bg-opacity-20 text-white border border-red-500 border-opacity-30'
                                                }`}>
                                                  <div className={`w-1.5 h-1.5 rounded-full ${placeDetailsCache[location.id].isOpen ? 'bg-green-400' : 'bg-red-400'}`}></div>
                                                  {placeDetailsCache[location.id].isOpen ? 'Open Now' : 'Closed'}
                                                </span>
                                              )}
                                              
                                              {placeDetailsCache[location.id].priceLevel && (
                                                <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-medium bg-orange-500 bg-opacity-20 text-orange-400 border border-orange-500 border-opacity-30">
                                                  {'$'.repeat(placeDetailsCache[location.id].priceLevel)}
                                                </span>
                                              )}
                                              
                                              {placeDetailsCache[location.id].types && placeDetailsCache[location.id].types.slice(0, 3).map((type: string, idx: number) => (
                                                <span key={idx} className="inline-flex items-center px-2 py-1 rounded-full text-[10px] font-medium bg-zinc-700 bg-opacity-40 text-stone-300 border border-zinc-600 border-opacity-30">
                                                  {type.replace(/_/g, ' ')}
                                                </span>
                                              ))}
                                            </div>
                                            
                                            {placeDetailsCache[location.id].reviews && placeDetailsCache[location.id].reviews.length > 0 && (
                                              <div className="pt-2 border-t border-zinc-700 border-opacity-40">
                                                <h4 className="text-white font-semibold text-xs mb-2 flex items-center gap-1.5">
                                                  <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                                  </svg>
                                                  User Reviews
                                                </h4>
                                                <div className="space-y-2.5">
                                                  {placeDetailsCache[location.id].reviews.slice(0, 3).map((review: any, idx: number) => (
                                                    <div key={idx} className="bg-zinc-800 bg-opacity-40 rounded-xl p-3 border border-zinc-700 border-opacity-20">
                                                      <div className="flex items-start justify-between mb-1.5">
                                                        <span className="text-white font-medium text-xs">{review.author}</span>
                                                        <div className="flex items-center gap-0.5">
                                                          {[...Array(5)].map((_, i) => (
                                                            <svg
                                                              key={i}
                                                              className={`w-3 h-3 ${i < review.rating ? 'text-yellow-400' : 'text-gray-600'}`}
                                                              fill="currentColor"
                                                              viewBox="0 0 20 20"
                                                            >
                                                              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                                            </svg>
                                                          ))}
                                                        </div>
                                                      </div>
                                                      <p className="text-stone-300 text-xs leading-relaxed mb-1">{review.text}</p>
                                                      {review.date && (
                                                        <span className="text-stone-500 text-[10px]">{review.date}</span>
                                                      )}
                                                    </div>
                                                  ))}
                                                  
                                                  {placeDetailsCache[location.id].reviews.length > 3 && (
                                                    <button className="w-full text-orange-400 hover:text-orange-300 text-xs font-medium py-2 transition-colors">
                                                      View All {placeDetailsCache[location.id].reviews.length} Reviews
                                                    </button>
                                                  )}
                                                </div>
                                              </div>
                                            )}
                                            
                                            {/* Action Links */}
                                            <div className="pt-3 mt-3 space-y-2">
                                              <button
                                                onClick={() => {
                                                  const address = placeDetailsCache[location.id]?.address || `${location.lat},${location.lng}`;
                                                  navigator.clipboard.writeText(address);
                                                  setCopiedItem(`sidebar-copy-${location.id}`);
                                                  setTimeout(() => setCopiedItem(null), 1000);
                                                }}
                                                className={`w-full text-left px-0 py-1.5 text-xs transition-colors flex items-center gap-2 ${
                                                  copiedItem === `sidebar-copy-${location.id}` ? 'text-orange-400' : 'text-stone-400 hover:text-white'
                                                }`}
                                              >
                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                                </svg>
                                                {copiedItem === `sidebar-copy-${location.id}` ? '✓ Copied!' : 'Copy Address'}
                                              </button>
                                              
                                              <a
                                                href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(location.name)}&query=${location.lat},${location.lng}`}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="block px-0 py-1.5 text-stone-400 hover:text-white text-xs transition-colors flex items-center gap-2"
                                              >
                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                                </svg>
                                                Open in Google Maps
                                              </a>
                                            </div>
                                          </div>
                                        ) : (
                                          <div className="py-4 text-center">
                                            <p className="text-stone-400 text-xs">Failed to load details</p>
                                          </div>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                </div>

                                {index < dayLocations.length - 1 && (
                                  <div className="flex justify-center -my-1 z-10 relative">
                                    {insertSearchInput[`${day}-${index}`] !== undefined ? (
                                      <div className="w-full px-4 py-2">
                                        <div className="relative">
                                          <input
                                            autoFocus
                                            type="text"
                                            value={insertSearchInput[`${day}-${index}`] || ''}
                                            placeholder="Search location..."
                                            spellCheck={false}
                                            autoComplete="off"
                                            className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-zinc-700 border-opacity-40 rounded-2xl text-white text-sm placeholder-white placeholder-opacity-60 focus:outline-none focus:border-orange-400 focus:bg-zinc-800 transition-all"
                                            onChange={(e) => handleInsertSearchChange(`${day}-${index}`, e.target.value)}
                                            onKeyDown={(e) => {
                                              if (e.key === 'Enter' && insertSearchInput[`${day}-${index}`]?.trim()) {
                                                handleAddPlace(day, insertSearchInput[`${day}-${index}`].trim(), index);
                                                const newInput = { ...insertSearchInput };
                                                delete newInput[`${day}-${index}`];
                                                setInsertSearchInput(newInput);
                                                const newShow = { ...showInsertSuggestions };
                                                delete newShow[`${day}-${index}`];
                                                setShowInsertSuggestions(newShow);
                                              } else if (e.key === 'Escape') {
                                                const newInput = { ...insertSearchInput };
                                                delete newInput[`${day}-${index}`];
                                                setInsertSearchInput(newInput);
                                              }
                                            }}
                                            onBlur={() => {
                                              setTimeout(() => {
                                                const newInput = { ...insertSearchInput };
                                                delete newInput[`${day}-${index}`];
                                                setInsertSearchInput(newInput);
                                              }, 200);
                                            }}
                                          />
                                          {showInsertSuggestions[`${day}-${index}`] && insertSearchSuggestions[`${day}-${index}`]?.length > 0 && (
                                            <div className="absolute top-full left-0 right-0 mt-1 z-[10000]">
                                              <ModernSearchSuggestions
                                                suggestions={insertSearchSuggestions[`${day}-${index}`]}
                                                onSelect={(suggestion) => {
                                                  handleAddPlace(day, suggestion, index);
                                                  const newInput = { ...insertSearchInput };
                                                  delete newInput[`${day}-${index}`];
                                                  setInsertSearchInput(newInput);
                                                  const newShow = { ...showInsertSuggestions };
                                                  delete newShow[`${day}-${index}`];
                                                  setShowInsertSuggestions(newShow);
                                                }}
                                              />
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    ) : (
                                      <button
                                        onClick={() => setInsertSearchInput({ ...insertSearchInput, [`${day}-${index}`]: '' })}
                                        className="w-7 h-7 bg-zinc-800 bg-opacity-60 hover:bg-opacity-80 backdrop-blur-xl rounded-full flex items-center justify-center transition-all border border-zinc-700 border-opacity-40 hover:border-orange-400 hover:scale-110 shadow-lg"
                                      >
                                        <span className="text-stone-400 hover:text-orange-400 transition-colors text-lg">+</span>
                                      </button>
                                    )}
                                  </div>
                                )}
                              </React.Fragment>
                            ))}

                            <div className="relative mt-3 mb-2">
                              <svg className="absolute left-4 top-1/2 transform -translate-y-1/2 w-4 h-4 text-white text-opacity-60 z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                              </svg>
                              <input
                                type="text"
                                value={bottomSearchInput[day] || ''}
                                placeholder="Add another place..."
                                spellCheck={false}
                                autoComplete="off"
                                onChange={(e) => handleBottomSearchChange(day, e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter' && bottomSearchInput[day] && bottomSearchInput[day].trim()) {
                                    handleAddPlace(day, bottomSearchInput[day].trim());
                                    setBottomSearchInput({ ...bottomSearchInput, [day]: '' });
                                    setShowBottomSuggestions({ ...showBottomSuggestions, [day]: false });
                                  }
                                }}
                                onFocus={async () => {
                                  const input = bottomSearchInput[day];
                                  if (input && input.trim().length > 1) {
                                    try {
                                      const countryParam = destination ? `&country=${encodeURIComponent(destination)}` : '';
                                      const response = await fetch('/api/landmarks/google-search?q=' + encodeURIComponent(input));
                                      const data = await response.json();
                                      
                                      const queryLower = input.toLowerCase();
                                      const suggestions = sortSuggestionsByRelevance(data.landmarks, queryLower).map((l: any) => l.name);
                                      setBottomSearchSuggestions({ ...bottomSearchSuggestions, [day]: suggestions.slice(0, 3) });
                                      setShowBottomSuggestions({ ...showBottomSuggestions, [day]: suggestions.length !== 0 });
                                    } catch (error) {
                                      console.error('Error searching landmarks:', error);
                                    }
                                  }
                                }}
                                className="w-full pl-10 pr-4 py-3 bg-zinc-800 bg-opacity-50 border border-zinc-700 border-opacity-40 rounded-2xl text-white text-sm placeholder-white placeholder-opacity-60 focus:outline-none focus:border-orange-400 focus:bg-zinc-800 transition-all"
                              />
                              {showBottomSuggestions[day] && bottomSearchSuggestions[day] && bottomSearchSuggestions[day].length > 0 ? (
                                <div className="absolute bottom-full left-0 right-0 mb-2 z-[10000]">
                                  <ModernSearchSuggestions
                                    suggestions={bottomSearchSuggestions[day]}
                                    onSelect={(suggestion) => {
                                      handleAddPlace(day, suggestion);
                                      setBottomSearchInput({ ...bottomSearchInput, [day]: '' });
                                      setShowBottomSuggestions({ ...showBottomSuggestions, [day]: false });
                                    }}
                                  />
                                </div>
                              ) : null}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          </div>
        </div>
      </div>

      {/* Modal for Maximized View */}
      {modalLocation && (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-transparent backdrop-blur-md animate-fadeIn" onClick={() => setModalLocation(null)}>
          <div className="bg-zinc-900 rounded-3xl shadow-2xl border border-zinc-700 w-[70%] max-w-3xl max-h-[75vh] overflow-hidden animate-scaleIn" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="relative h-48 bg-gradient-to-br from-zinc-800 to-zinc-900 border-b border-zinc-700">
              {modalLocation.image && (
                <Image
                  src={modalLocation.image}
                  alt={modalLocation.name}
                  fill
                  quality={95}
                  className="object-cover opacity-60"
                />
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-zinc-900 via-transparent to-transparent" />
              <div className="absolute bottom-4 left-6 right-6">
                <h2 className="text-3xl font-bold text-white drop-shadow-lg">{modalLocation.name}</h2>
                <div className="flex items-center gap-2 mt-2">
                  <span className="px-3 py-1 bg-orange-500 bg-opacity-20 backdrop-blur-sm text-white text-xs font-medium rounded-full border border-orange-500 border-opacity-30">
                    Day {modalLocation.day}
                  </span>
                  {modalLocation.confidence && (
                    <span className="px-3 py-1 bg-zinc-800 bg-opacity-80 backdrop-blur-sm text-white text-xs font-medium rounded-full">
                      {(modalLocation.confidence * 100).toFixed(0)}% Match
                    </span>
                  )}
                </div>
              </div>
              <button
                onClick={() => setModalLocation(null)}
                className="absolute top-4 right-4 w-10 h-10 bg-zinc-800 bg-opacity-80 hover:bg-opacity-100 backdrop-blur-sm rounded-full flex items-center justify-center transition-all hover:scale-110"
              >
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto max-h-[calc(75vh-12rem)]">
              {loadingPlaceDetails === modalLocation.id ? (
                <div className="py-12 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-orange-500"></div>
                </div>
              ) : placeDetailsCache[modalLocation.id] ? (
                <div className="space-y-6">
                  {placeDetailsCache[modalLocation.id].rating && (
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1">
                        {[...Array(5)].map((_, i) => (
                          <svg
                            key={i}
                            className={`w-6 h-6 ${i < Math.floor(placeDetailsCache[modalLocation.id].rating) ? 'text-yellow-400' : 'text-gray-600'}`}
                            fill="currentColor"
                            viewBox="0 0 20 20"
                          >
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                        ))}
                      </div>
                      <span className="text-white font-semibold text-lg">{placeDetailsCache[modalLocation.id].rating.toFixed(1)}</span>
                      {placeDetailsCache[modalLocation.id].totalRatings && (
                        <span className="text-stone-400 text-sm">({placeDetailsCache[modalLocation.id].totalRatings.toLocaleString()} reviews)</span>
                      )}
                    </div>
                  )}

                  {placeDetailsCache[modalLocation.id].photos && placeDetailsCache[modalLocation.id].photos.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-white font-semibold text-lg flex items-center gap-2">
                        <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        Photos
                      </h3>
                      <div className="grid grid-cols-3 gap-3">
                        {placeDetailsCache[modalLocation.id].photos.map((photoUrl: string, idx: number) => (
                          <div 
                            key={idx} 
                            className="aspect-video rounded-xl overflow-hidden bg-zinc-800 border border-zinc-700 cursor-pointer"
                            onClick={() => {
                              setCurrentPhotos(placeDetailsCache[modalLocation.id].photos);
                              setSelectedPhotoIndex(idx);
                            }}
                          >
                            <Image
                              src={photoUrl}
                              alt={`${modalLocation.name} photo ${idx + 1}`}
                              width={400}
                              height={300}
                              className="w-full h-full object-cover hover:scale-110 transition-transform duration-300"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {placeDetailsCache[modalLocation.id].description && (
                    <div>
                      <p className="text-stone-300 text-base leading-relaxed">{placeDetailsCache[modalLocation.id].description}</p>
                    </div>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {placeDetailsCache[modalLocation.id].openingHours && (
                      <div className="flex items-start gap-3 p-4 bg-zinc-800 bg-opacity-50 rounded-xl border border-zinc-700 border-opacity-40">
                        <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <div>
                          <h4 className="text-white font-medium text-sm mb-1">Hours</h4>
                          <p className="text-stone-300 text-sm whitespace-pre-line">{placeDetailsCache[modalLocation.id].openingHours}</p>
                        </div>
                      </div>
                    )}

                    {placeDetailsCache[modalLocation.id].address && (
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(placeDetailsCache[modalLocation.id].address);
                          setCopiedItem('address');
                          setTimeout(() => setCopiedItem(null), 1000);
                        }}
                        className={`flex items-start gap-3 p-4 rounded-xl transition-all duration-300 cursor-pointer w-full text-left group ${
                          copiedItem === 'address'
                            ? 'bg-zinc-800 bg-opacity-50 border-2 border-orange-500'
                            : 'bg-zinc-800 bg-opacity-50 border border-zinc-700 border-opacity-40 hover:scale-105'
                        }`}
                      >
                        <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <div className="flex-1">
                          <h4 className="text-white font-medium text-sm mb-1 flex items-center gap-2">
                            Address
                            {copiedItem === 'address' && (
                              <span className="text-green-400 text-xs animate-fadeIn">✓ Copied!</span>
                            )}
                          </h4>
                          <p className="text-stone-300 text-sm">{placeDetailsCache[modalLocation.id].address}</p>
                        </div>
                        <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                      </button>
                    )}

                    {placeDetailsCache[modalLocation.id].phone && (
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(placeDetailsCache[modalLocation.id].phone);
                          setCopiedItem('phone');
                          setTimeout(() => setCopiedItem(null), 1000);
                        }}
                        className={`flex items-start gap-3 p-4 rounded-xl transition-all duration-300 cursor-pointer w-full text-left group ${
                          copiedItem === 'phone'
                            ? 'bg-zinc-800 bg-opacity-50 border-2 border-orange-500'
                            : 'bg-zinc-800 bg-opacity-50 border border-zinc-700 border-opacity-40 hover:scale-105'
                        }`}
                      >
                        <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                        </svg>
                        <div className="flex-1">
                          <h4 className="text-white font-medium text-sm mb-1 flex items-center gap-2">
                            Phone
                            {copiedItem === 'phone' && (
                              <span className="text-green-400 text-xs animate-fadeIn">✓ Copied!</span>
                            )}
                          </h4>
                          <p className="text-stone-300 text-sm">{placeDetailsCache[modalLocation.id].phone}</p>
                        </div>
                        <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                      </button>
                    )}

                    {placeDetailsCache[modalLocation.id].website && (
                      <div className="flex items-start gap-3 p-4 bg-zinc-800 bg-opacity-50 rounded-xl border border-zinc-700 border-opacity-40">
                        <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                        </svg>
                        <div>
                          <h4 className="text-white font-medium text-sm mb-1">Website</h4>
                          <a
                            href={placeDetailsCache[modalLocation.id].website}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-orange-400 hover:text-orange-300 text-sm transition-colors break-all"
                          >
                            Visit Website
                          </a>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {placeDetailsCache[modalLocation.id].isOpen !== undefined && (
                      <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${
                        placeDetailsCache[modalLocation.id].isOpen 
                          ? 'bg-green-500 bg-opacity-20 text-white border border-green-500 border-opacity-30' 
                          : 'bg-red-500 bg-opacity-20 text-white border border-red-500 border-opacity-30'
                      }`}>
                        <div className={`w-2 h-2 rounded-full ${placeDetailsCache[modalLocation.id].isOpen ? 'bg-green-400' : 'bg-red-400'}`}></div>
                        {placeDetailsCache[modalLocation.id].isOpen ? 'Open Now' : 'Closed'}
                      </span>
                    )}

                    {placeDetailsCache[modalLocation.id].priceLevel && (
                      <span className="inline-flex items-center px-3 py-1.5 rounded-full text-sm font-medium bg-orange-500 bg-opacity-20 text-orange-400 border border-orange-500 border-opacity-30">
                        {'$'.repeat(placeDetailsCache[modalLocation.id].priceLevel)}
                      </span>
                    )}

                    {placeDetailsCache[modalLocation.id].types && placeDetailsCache[modalLocation.id].types.slice(0, 5).map((type: string, idx: number) => (
                      <span key={idx} className="inline-flex items-center px-3 py-1.5 rounded-full text-sm font-medium bg-zinc-700 bg-opacity-40 text-stone-300 border border-zinc-600 border-opacity-30">
                        {type.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>

                  {placeDetailsCache[modalLocation.id].reviews && placeDetailsCache[modalLocation.id].reviews.length > 0 && (
                    <div className="pt-4 border-t border-zinc-700 border-opacity-40">
                      <h3 className="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                        <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                        User Reviews
                      </h3>
                      <div className="space-y-4">
                        {placeDetailsCache[modalLocation.id].reviews.map((review: any, idx: number) => (
                          <div key={idx} className="bg-zinc-800 bg-opacity-40 rounded-xl p-4 border border-zinc-700 border-opacity-20">
                            <div className="flex items-start justify-between mb-2">
                              <span className="text-white font-medium text-sm">{review.author}</span>
                              <div className="flex items-center gap-1">
                                {[...Array(5)].map((_, i) => (
                                  <svg
                                    key={i}
                                    className={`w-4 h-4 ${i < review.rating ? 'text-yellow-400' : 'text-gray-600'}`}
                                    fill="currentColor"
                                    viewBox="0 0 20 20"
                                  >
                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                  </svg>
                                ))}
                              </div>
                            </div>
                            <p className="text-stone-300 text-sm leading-relaxed mb-2">{review.text}</p>
                            {review.date && (
                              <span className="text-stone-500 text-xs">{review.date}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex gap-3 pt-4">
                    <button
                      onClick={() => {
                        const address = placeDetailsCache[modalLocation.id]?.address || `${modalLocation.lat},${modalLocation.lng}`;
                        navigator.clipboard.writeText(address);
                        setCopiedItem('copyButton');
                        setTimeout(() => setCopiedItem(null), 1000);
                      }}
                      className="flex-1 px-4 py-3 bg-zinc-800 hover:bg-zinc-700 text-white text-sm font-medium rounded-xl transition-all flex items-center justify-center gap-2 hover:scale-105"
                    >
                      {copiedItem === 'copyButton' ? (
                        <>
                          <svg className="w-5 h-5 text-green-400 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          <span className="text-green-400">Copied!</span>
                        </>
                      ) : (
                        <>
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                          Copy Address
                        </>
                      )}
                    </button>
                    <a
                      href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(modalLocation.name)}&query=${modalLocation.lat},${modalLocation.lng}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex-1 px-4 py-3 bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white text-sm font-medium rounded-xl transition-all flex items-center justify-center gap-2"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                      Open in Google Maps
                    </a>
                  </div>
                </div>
              ) : (
                <div className="py-12 text-center">
                  <p className="text-stone-400 text-base">Failed to load details</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Photo Viewer Modal */}
      {selectedPhotoIndex !== null && currentPhotos.length > 0 && (
        <div 
          className="fixed inset-0 z-[20000] flex items-center justify-center bg-black bg-opacity-90 backdrop-blur-sm" 
          onClick={() => {
            setSelectedPhotoIndex(null);
            setCurrentPhotos([]);
          }}
        >
          <div className="relative max-w-7xl max-h-[90vh] w-full px-4">
            {/* Close Button */}
            <button
              onClick={() => {
                setSelectedPhotoIndex(null);
                setCurrentPhotos([]);
              }}
              className="absolute top-4 right-4 w-12 h-12 bg-zinc-800 bg-opacity-80 hover:bg-opacity-100 backdrop-blur-sm rounded-full flex items-center justify-center transition-all hover:scale-110 z-10"
            >
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            
            {/* Left Arrow */}
            {currentPhotos.length > 1 && selectedPhotoIndex > 0 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedPhotoIndex(selectedPhotoIndex - 1);
                }}
                className="absolute left-4 top-1/2 -translate-y-1/2 w-12 h-12 bg-zinc-800 bg-opacity-80 hover:bg-opacity-100 backdrop-blur-sm rounded-full flex items-center justify-center transition-all hover:scale-110 z-10"
              >
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
            )}
            
            {/* Right Arrow */}
            {currentPhotos.length > 1 && selectedPhotoIndex < currentPhotos.length - 1 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedPhotoIndex(selectedPhotoIndex + 1);
                }}
                className="absolute right-4 top-1/2 -translate-y-1/2 w-12 h-12 bg-zinc-800 bg-opacity-80 hover:bg-opacity-100 backdrop-blur-sm rounded-full flex items-center justify-center transition-all hover:scale-110 z-10"
              >
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            )}
            
            <div className="relative w-full h-full flex items-center justify-center" onClick={(e) => e.stopPropagation()}>
              <Image
                src={currentPhotos[selectedPhotoIndex]}
                alt={`Photo ${selectedPhotoIndex + 1} of ${currentPhotos.length}`}
                width={1200}
                height={800}
                quality={100}
                className="max-w-full max-h-[90vh] w-auto h-auto object-contain rounded-2xl"
              />
            </div>
            
            {/* Photo Counter */}
            {currentPhotos.length > 1 && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 bg-zinc-800 bg-opacity-80 backdrop-blur-sm rounded-full text-white text-sm font-medium">
                {selectedPhotoIndex + 1} / {currentPhotos.length}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Share Modal */}
      {showShareModal && (
        <div 
          className="fixed inset-0 z-[10000] flex items-center justify-center bg-transparent backdrop-blur-md"
          onClick={() => setShowShareModal(false)}
        >
          <div 
            className="relative bg-zinc-900 rounded-2xl shadow-2xl w-full max-w-md p-6 animate-scaleIn border border-zinc-800"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={() => setShowShareModal(false)}
              className="absolute top-4 right-4 text-stone-400 hover:text-white transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Modal Content */}
            <div className="space-y-4">
              <h3 className="text-2xl font-bold text-white">Share Your Itinerary</h3>
              <p className="text-stone-400 text-sm">Copy the link below to share your travel plans with others. They'll be able to view your itinerary in read-only mode.</p>
              
              {/* Shareable Link Display */}
              <div className="bg-zinc-800 rounded-lg p-4 border border-zinc-700">
                <div className="flex items-center gap-3">
                  <div className="flex-1 overflow-hidden">
                    <p className="text-xs text-stone-500 mb-1">Shareable Link</p>
                    <p className="text-white text-sm font-mono truncate">{shareableLink}</p>
                  </div>
                  <button
                    onClick={handleCopyShareLink}
                    className={`px-4 py-2 rounded-lg transition-all ${
                      copiedItem === 'shareLink' 
                        ? 'bg-orange-500 text-white animate-bounce' 
                        : 'bg-orange-600 hover:bg-orange-500 text-white'
                    }`}
                  >
                    {copiedItem === 'shareLink' ? (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>

              {/* Info Note */}
              <div className="bg-zinc-800 bg-opacity-50 rounded-lg p-3 border border-zinc-700 border-opacity-50">
                <div className="flex gap-2">
                  <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-stone-400 text-xs">This link will remain valid as long as your browser storage is not cleared. Anyone with this link can view your itinerary.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Prediction Modal */}
      {predictionModal?.show && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999] p-4">
          <div className="bg-gradient-to-br from-zinc-900 to-stone-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-orange-500/20">
            {/* Header */}
            <div className="sticky top-0 bg-gradient-to-r from-zinc-900 to-stone-900 border-b border-orange-500/20 p-6 z-10">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-orange-400">Landmark Detected</h2>
                  <p className="text-stone-400 text-sm mt-1">Select the correct landmark or request better analysis</p>
                </div>
                <button
                  onClick={() => setPredictionModal(null)}
                  className="text-stone-400 hover:text-orange-400 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="p-6 space-y-6">
              {/* Google Vision Status Banner */}
              {predictionModal.predictions[0]?.confidence < 0.70 && (
                <div className={`border-2 rounded-xl p-4 ${
                  predictionModal.googleVisionResult 
                    ? 'bg-gradient-to-r from-blue-900/20 via-green-900/20 to-blue-900/20 border-blue-500/30' 
                    : 'bg-gradient-to-r from-yellow-900/20 to-orange-900/20 border-yellow-500/30'
                }`}>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0">
                      {predictionModal.googleVisionResult ? (
                        <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      ) : (
                        <svg className="w-8 h-8 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1">
                      {predictionModal.googleVisionResult ? (
                        <>
                          <div className="font-semibold text-blue-300 text-sm">✓ Google Vision Validation</div>
                          <p className="text-stone-400 text-xs mt-0.5">
                            Confirmed: <span className="text-green-400 font-medium">{predictionModal.googleVisionResult.landmark_name}</span>
                            {' '}({(predictionModal.googleVisionResult.confidence * 100).toFixed(1)}% confidence) - See top result below
                          </p>
                        </>
                      ) : (
                        <>
                          <div className="font-semibold text-yellow-300 text-sm">⚠ Low Confidence Detection</div>
                          <p className="text-stone-400 text-xs mt-0.5">
                            Google Vision API consulted but found no famous landmarks. Consider using Tier 2 if result seems incorrect.
                          </p>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}
              
              {/* Image Preview */}
              <div className="relative w-full h-80 bg-zinc-800 rounded-xl overflow-hidden">
                <img
                  src={predictionModal.imagePreview}
                  alt="Uploaded landmark"
                  className="w-full h-full object-contain"
                />
              </div>

              {/* Predictions List */}
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-stone-100">Top Predictions:</h3>
                
                {/* Google Vision Result - Show First if Available */}
                {predictionModal.googleVisionResult && (
                  <button
                    onClick={async () => {
                      const result = predictionModal.googleVisionResult!;
                      const placeData = await fetchPlacePhotos(result.landmark_name);
                      const lat = result.locations[0]?.latitude || destinationLat;
                      const lng = result.locations[0]?.longitude || destinationLng;
                      const imageUrl = placeData.photos && placeData.photos.length > 0 ? placeData.photos[0].url : '';
                      
                      const newLocation: Location = {
                        id: Date.now().toString(),
                        name: result.landmark_name,
                        lat,
                        lng,
                        image: imageUrl,
                        confidence: result.confidence,
                        day: currentDay,
                      };
                      
                      setLocations([...locations, newLocation]);
                      setPredictionModal(null);
                    }}
                    className="w-full bg-zinc-800/50 hover:bg-zinc-800 border-2 border-blue-500/50 hover:border-blue-400 rounded-xl overflow-hidden transition-all duration-200 text-left group"
                  >
                    {/* Image Gallery */}
                    {predictionModal.googleVisionResult.photos && predictionModal.googleVisionResult.photos.length > 0 ? (
                      <div className="flex overflow-x-auto gap-2 p-3 bg-zinc-900/50">
                        {predictionModal.googleVisionResult.photos.map((photo: any, imgIdx: number) => (
                          <img
                            key={imgIdx}
                            src={photo.url}
                            alt={`${predictionModal.googleVisionResult.landmark_name} ${imgIdx + 1}`}
                            className="h-32 w-48 object-cover rounded flex-shrink-0"
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="flex overflow-x-auto gap-2 p-3 bg-zinc-900/50">
                        {[1, 2, 3].map((imgIdx) => (
                          <div
                            key={imgIdx}
                            className="h-32 w-48 bg-gradient-to-br from-zinc-800 to-zinc-900 rounded flex-shrink-0 flex items-center justify-center text-stone-600"
                          >
                            <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* Prediction Info */}
                    <div className="p-5">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <span className="text-blue-400 font-bold text-xl">#1</span>
                            <div>
                              <div className="flex items-center gap-2">
                                <div className="text-stone-100 font-semibold text-lg group-hover:text-blue-400 transition-colors">
                                  {predictionModal.googleVisionResult.landmark_name}
                                </div>
                                <span className="px-2 py-0.5 bg-blue-500/20 border border-blue-400/30 rounded text-blue-300 text-xs font-semibold">
                                  GOOGLE VISION
                                </span>
                              </div>
                              {(predictionModal.googleVisionResult.city || predictionModal.googleVisionResult.country) && (
                                <p className="text-stone-500 text-sm mt-1">
                                  {predictionModal.googleVisionResult.city}{predictionModal.googleVisionResult.city && predictionModal.googleVisionResult.country ? ', ' : ''}{predictionModal.googleVisionResult.country}
                                </p>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <div className="text-blue-400 font-bold text-lg">
                              {(predictionModal.googleVisionResult.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="text-stone-500 text-xs">confidence</div>
                          </div>
                          <svg className="w-5 h-5 text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  </button>
                )}
                
                {predictionModal.predictions.slice(0, 4).map((pred, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAcceptPrediction(pred)}
                    className="w-full bg-zinc-800/50 hover:bg-zinc-800 border border-stone-700 hover:border-orange-500/50 rounded-xl overflow-hidden transition-all duration-200 text-left group"
                  >
                    {/* Image Gallery */}
                    {pred.photos && pred.photos.length > 0 ? (
                      <div className="flex overflow-x-auto gap-2 p-3 bg-zinc-900/50">
                        {pred.photos.map((photo: any, imgIdx: number) => (
                          <img
                            key={imgIdx}
                            src={photo.url}
                            alt={`${pred.landmark} ${imgIdx + 1}`}
                            className="h-32 w-48 object-cover rounded flex-shrink-0"
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="flex overflow-x-auto gap-2 p-3 bg-zinc-900/50">
                        {[1, 2, 3].map((imgIdx) => (
                          <div
                            key={imgIdx}
                            className="h-32 w-48 bg-gradient-to-br from-zinc-800 to-zinc-900 rounded flex-shrink-0 flex items-center justify-center text-stone-600"
                          >
                            <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* Prediction Info */}
                    <div className="p-5">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <span className="text-orange-400 font-bold text-xl">#{predictionModal.googleVisionResult ? idx + 2 : idx + 1}</span>
                            <div>
                              <div className="text-stone-100 font-semibold text-lg group-hover:text-orange-400 transition-colors">
                                {pred.landmark}
                              </div>
                              {(pred.city || pred.country) && (
                                <p className="text-stone-500 text-sm mt-1">
                                  {pred.city}{pred.city && pred.country ? ', ' : ''}{pred.country}
                                </p>
                              )}
                              {pred.latitude && pred.longitude && (
                                <p className="text-stone-400 text-xs mt-0.5">
                                  📍 {pred.latitude.toFixed(4)}, {pred.longitude.toFixed(4)}
                                </p>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <div className="text-orange-400 font-bold text-lg">
                              {(pred.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="text-stone-500 text-xs">confidence</div>
                          </div>
                          <svg className="w-5 h-5 text-orange-400 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              {/* Fallback Button */}
              <div className="pt-4 border-t border-stone-700">
                <button
                  onClick={handleRejectPrediction}
                  disabled={loadingFallback}
                  className="w-full bg-zinc-800/50 hover:bg-zinc-800 border-2 border-purple-500 hover:border-purple-400 disabled:opacity-50 text-purple-300 hover:text-purple-200 font-semibold py-4 px-6 rounded-xl transition-all duration-200 flex items-center justify-center gap-2"
                >
                  <span>None of These? Analyze Deeper with AI</span>
                </button>
                <p className="text-stone-500 text-sm text-center mt-2">
                  Uses CLIP visual similarity + Groq vision AI for deeper analysis
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[9999] p-4">
          <div className="bg-gradient-to-br from-zinc-900 to-black border border-stone-700 rounded-2xl shadow-2xl max-w-lg w-full overflow-hidden">
            <div className="p-6">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-stone-100">Upload Your Photos</h2>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="text-stone-400 hover:text-stone-200 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {!uploading ? (
                <>
                  {/* Drag and Drop Area */}
                  <label
                    htmlFor="modal-file-upload"
                    className="block border-2 border-dashed border-orange-500/50 hover:border-orange-500 rounded-xl p-12 text-center cursor-pointer transition-all hover:bg-orange-500/5 group"
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.currentTarget.classList.add('border-orange-500', 'bg-orange-500/10');
                    }}
                    onDragLeave={(e) => {
                      e.currentTarget.classList.remove('border-orange-500', 'bg-orange-500/10');
                    }}
                    onDrop={(e) => {
                      e.preventDefault();
                      e.currentTarget.classList.remove('border-orange-500', 'bg-orange-500/10');
                      const files = e.dataTransfer.files;
                      if (files && files.length > 0) {
                        const input = document.getElementById('modal-file-upload') as HTMLInputElement;
                        if (input) {
                          const dataTransfer = new DataTransfer();
                          for (let i = 0; i < files.length; i++) {
                            dataTransfer.items.add(files[i]);
                          }
                          input.files = dataTransfer.files;
                          handleFileUpload({ target: input } as any);
                        }
                      }
                    }}
                  >
                    <div className="flex flex-col items-center gap-4">
                      <div className="w-16 h-16 rounded-full bg-orange-500/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                        <svg className="w-8 h-8 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                      </div>
                      <div>
                        <p className="text-stone-100 font-medium mb-1">
                          Click to browse or drag and drop
                        </p>
                        <p className="text-stone-500 text-sm">
                          Upload photos of landmarks to detect
                        </p>
                      </div>
                    </div>
                  </label>
                  <input
                    type="file"
                    id="modal-file-upload"
                    multiple
                    accept="image/*"
                    onChange={(e) => handleFileUpload(e)}
                    className="hidden"
                  />

                  {/* Upload Button */}
                  <button
                    onClick={() => document.getElementById('modal-file-upload')?.click()}
                    className="w-full mt-4 bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white font-semibold py-3 px-6 rounded-xl transition-all hover:scale-[1.02] hover:shadow-lg flex items-center justify-center gap-2"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Upload Photos
                  </button>
                </>
              ) : (
                <>
                  {/* AI Loading Animation */}
                  <div className="flex flex-col items-center justify-center py-12">
                    <div className="relative w-24 h-24 mb-6">
                      {/* Outer rotating ring */}
                      <div className="absolute inset-0 rounded-full border-4 border-orange-500/20"></div>
                      <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-orange-500 animate-spin"></div>
                      
                      {/* Middle pulsing ring */}
                      <div className="absolute inset-2 rounded-full border-4 border-orange-400/30 animate-pulse"></div>
                      
                      {/* Inner rotating ring (reverse) */}
                      <div className="absolute inset-4 rounded-full border-4 border-transparent border-b-orange-300 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
                      
                      {/* Center AI icon */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <svg className="w-10 h-10 text-orange-500 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                          <circle cx="8" cy="9" r="1" fill="currentColor" />
                          <circle cx="12" cy="9" r="1" fill="currentColor" />
                          <circle cx="16" cy="9" r="1" fill="currentColor" />
                        </svg>
                      </div>
                    </div>

                    {/* Loading text with gradient */}
                    <h3 className="text-xl font-bold bg-gradient-to-r from-orange-400 via-orange-500 to-orange-600 bg-clip-text text-transparent mb-2">
                      Analyzing with AI
                    </h3>
                    <p className="text-stone-400 text-sm">
                      Detecting landmarks in your photos...
                    </p>

                    {/* Progress dots */}
                    <div className="flex gap-2 mt-6">
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Purple Fallback Modal - "Thinking Harder" */}
      {showFallbackModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[10000] p-4">
          <div className="bg-gradient-to-br from-purple-900/50 to-black border border-purple-700/50 rounded-2xl shadow-2xl max-w-md w-full overflow-hidden">
            <div className="p-12 flex flex-col items-center justify-center space-y-6">
              {/* Triple rotating rings - larger */}
              <div className="relative w-24 h-24">
                {/* Outer rotating ring */}
                <div className="absolute inset-0 rounded-full border-4 border-purple-500/20"></div>
                <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-purple-500 animate-spin"></div>
                
                {/* Middle pulsing ring */}
                <div className="absolute inset-3 rounded-full border-4 border-purple-400/30 animate-pulse"></div>
                
                {/* Inner rotating ring (reverse, faster) */}
                <div className="absolute inset-6 rounded-full border-4 border-transparent border-b-purple-300 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1s' }}></div>
                
                {/* Center brain/thinking icon */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <svg className="w-10 h-10 text-purple-400 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
              </div>

              {/* Loading text with gradient */}
              <div className="text-center">
                <h3 className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-purple-500 to-purple-600 bg-clip-text text-transparent mb-3">
                  Thinking Harder...
                </h3>
                <p className="text-purple-300/80 text-sm">
                  Using CLIP + Groq AI for deeper analysis
                </p>
              </div>

              {/* Progress dots */}
              <div className="flex gap-2">
                <div className="w-2.5 h-2.5 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                <div className="w-2.5 h-2.5 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2.5 h-2.5 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Fallback Confirmation Modal */}
      {fallbackConfirmModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[10001] p-4">
          <div className="bg-gradient-to-br from-zinc-900 to-black border border-purple-700/50 rounded-2xl shadow-2xl max-w-2xl w-full overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-purple-900/30 to-purple-800/20 border-b border-purple-700/30 px-6 py-4">
              <h2 className="text-2xl font-bold text-stone-100">AI Analysis Result</h2>
              <p className="text-purple-300 text-sm mt-1">Confirm if this matches your photo</p>
            </div>

            <div className="p-6 space-y-6">
              {/* Landmark Name */}
              <div className="text-center">
                <h3 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent">
                  {fallbackConfirmModal.landmarkName}
                </h3>
                {fallbackConfirmModal.confidence && !isNaN(fallbackConfirmModal.confidence) && (
                  <div className="flex items-center justify-center gap-2 mt-2">
                    <div className="px-3 py-1 bg-purple-500/20 border border-purple-500/30 rounded-full">
                      <span className="text-purple-300 text-sm font-medium">
                        {(fallbackConfirmModal.confidence * 100).toFixed(1)}% Match
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Image */}
              {fallbackConfirmModal.image && (
                <div className="relative w-full h-80 bg-zinc-800 rounded-xl overflow-hidden">
                  <img
                    src={fallbackConfirmModal.image}
                    alt={fallbackConfirmModal.landmarkName}
                    className="w-full h-full object-cover"
                  />
                </div>
              )}

              {/* AI Analysis - Formatted */}
              <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/10 border border-purple-700/30 rounded-xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <h4 className="text-lg font-semibold text-purple-300">AI Vision Analysis</h4>
                </div>
                <div className="text-stone-300 leading-loose text-base space-y-3">
                  {fallbackConfirmModal.visionDescription.split('\n\n').map((paragraph, idx) => {
                    // Check if paragraph is a bullet point list
                    if (paragraph.includes('\n- ') || paragraph.startsWith('- ')) {
                      const items = paragraph.split('\n').filter(line => line.trim());
                      return (
                        <ul key={idx} className="list-disc list-inside space-y-2 pl-2">
                          {items.map((item, itemIdx) => (
                            <li key={itemIdx} className="text-stone-300">
                              {item.replace(/^-\s*/, '')}
                            </li>
                          ))}
                        </ul>
                      );
                    }
                    // Check if paragraph has numbered list
                    else if (paragraph.match(/^\d+\./)) {
                      const items = paragraph.split('\n').filter(line => line.trim());
                      return (
                        <ol key={idx} className="list-decimal list-inside space-y-2 pl-2">
                          {items.map((item, itemIdx) => (
                            <li key={itemIdx} className="text-stone-300">
                              {item.replace(/^\d+\.\s*/, '')}
                            </li>
                          ))}
                        </ol>
                      );
                    }
                    // Regular paragraph
                    return (
                      <p key={idx} className="text-justify">
                        {paragraph}
                      </p>
                    );
                  })}
                </div>
              </div>

              {/* Buttons */}
              <div className="flex gap-3 pt-2">
                <button
                  onClick={handleRejectFallbackLocation}
                  className="flex-1 bg-zinc-800/50 hover:bg-zinc-800 border border-stone-700 hover:border-red-500/50 text-stone-300 hover:text-red-400 font-semibold py-4 px-6 rounded-xl transition-all duration-200"
                >
                  No, This Isn't Right
                </button>
                <button
                  onClick={handleConfirmFallbackLocation}
                  className="flex-1 bg-zinc-800/50 hover:bg-zinc-800 border-2 border-purple-500 hover:border-purple-400 text-purple-300 hover:text-purple-200 font-semibold py-4 px-6 rounded-xl transition-all duration-200"
                >
                  Yes, Add to Itinerary
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

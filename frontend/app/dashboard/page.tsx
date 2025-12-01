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
    if (lat) setDestinationLat(parseFloat(lat));
    if (lng) setDestinationLng(parseFloat(lng));

    if (dest) {
      const destLower = dest.toLowerCase();
      if (DESTINATION_IMAGES[destLower]) {
        setDestinationImage(DESTINATION_IMAGES[destLower]);
      }
    }
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    
    console.log('Uploading files:', files);
    
    setTimeout(() => {
      setUploading(false);
      const newLocation: Location = {
        id: Date.now().toString(),
        name: 'Detected Landmark',
        lat: 35.6762 + Math.random() * 0.1,
        lng: 139.6503 + Math.random() * 0.1,
        confidence: 0.8 + Math.random() * 0.15,
        day: currentDay,
      };
      setLocations([...locations, newLocation]);
    }, 2000);
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
      setPlaceDetailsCache(prev => ({ ...prev, [locationId]: data }));
    } catch (error) {
      console.error('Error fetching place details:', error);
      setPlaceDetailsCache(prev => ({ ...prev, [locationId]: null }));
    } finally {
      setLoadingPlaceDetails(null);
    }
  };

  const getLocationImage = async (locationName: string, lat?: number, lng?: number): Promise<string> => {
    // Try to fetch image from API first (with Unsplash integration)
    if (lat && lng) {
      try {
        const response = await fetch(`/api/place-details?name=${encodeURIComponent(locationName)}&lat=${lat}&lng=${lng}`);
        const data = await response.json();
        if (data.image) {
          return data.image;
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
    
    try {
      const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(placeName));
      const data = await response.json();
      if (data.landmarks && data.landmarks.length !== 0 && data.landmarks[0].latitude && data.landmarks[0].longitude) {
        lat = data.landmarks[0].latitude;
        lng = data.landmarks[0].longitude;
      }
    } catch (error) {
      console.error('Error fetching coordinates:', error);
    }
    
    const imageUrl = await getLocationImage(placeName, lat, lng);
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
    setLoadingRecommendations(true);
    
    setTimeout(async () => {
      const image1 = await getLocationImage('Senso-ji Temple Tokyo');
      const image2 = await getLocationImage('Meiji Shrine Tokyo');
      
      const mockLocations: Location[] = [
        {
          id: Date.now().toString() + '-1',
          name: 'Senso-ji Temple',
          lat: 35.7148,
          lng: 139.7967,
          image: image1,
          day: day,
        },
        {
          id: Date.now().toString() + '-2',
          name: 'Meiji Shrine',
          lat: 35.6764,
          lng: 139.6993,
          image: image2,
          day: day,
        },
      ];
      setLocations([...locations, ...mockLocations]);
      setLoadingRecommendations(false);
    }, 1500);
  };

  const handleUpdateNote = (locationId: string, note: string) => {
    setLocations(locations.map(loc => 
      loc.id === locationId ? { ...loc, notes: note } : loc
    ));
  };

  const handleSearchChange = (value: string) => {
    setSearchInput(value);
    
    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
    }
    
    if (value.trim().length > 1) {
      searchDebounceRef.current = setTimeout(async () => {
        try {
          const countryParam = destination ? `&country=${encodeURIComponent(destination)}` : '';
          const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(value) + countryParam);
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
          const countryParam = destination ? `&country=${encodeURIComponent(destination)}` : '';
          const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(value) + countryParam);
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
          const countryParam = destination ? `&country=${encodeURIComponent(destination)}` : '';
          const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(value) + countryParam);
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
          const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(value));
          const data = await response.json();
          // Filter to only show large locations (countries, states, regions) - NO cities or smaller
          const suggestions = data.landmarks
            .filter((l: any) => {
              const addresstype = (l.addresstype || '').toLowerCase();
              const type = (l.type || '').toLowerCase();
              
              // Only allow: country, state, region (administrative level 2-4)
              const isLargeLocation = addresstype === 'country' || 
                                     addresstype === 'state' || 
                                     addresstype === 'region' ||
                                     type === 'administrative';
              
              // Exclude cities, towns, villages, and smaller
              const isSmallLocation = addresstype === 'city' || 
                                     addresstype === 'town' || 
                                     addresstype === 'village' ||
                                     addresstype === 'municipality' ||
                                     addresstype === 'county' ||
                                     type === 'city';
              
              return isLargeLocation && !isSmallLocation;
            })
            .map((l: any) => l.name);
          setDestinationSuggestions(suggestions.slice(0, 3));
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
      const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(dest));
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
          <button className="text-stone-400 hover:text-stone-300 px-3 py-1.5 transition-colors text-sm">
            Share
          </button>
        </div>
      </div>

      <div className="flex-1 relative">
        <div className="absolute inset-0">
          <MapComponent landmarks={locations} selectedLandmark={selectedLocation} />
        </div>

        <div className="absolute top-5 left-5 bottom-5 w-96 min-w-[384px] bg-zinc-900 bg-opacity-95 backdrop-blur-3xl border border-zinc-800 border-opacity-50 flex flex-col shadow-[0_8px_32px_0_rgba(0,0,0,0.6)] rounded-3xl z-[1000]" style={{overflow: 'visible'}}>
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
                        <div className="w-9 h-9 rounded-2xl bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center text-white text-sm font-bold shadow-lg flex-shrink-0">
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
                                  const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(searchInput) + countryParam);
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
                            className="bg-gradient-to-br from-purple-500 to-purple-600 text-white px-4 py-2.5 rounded-2xl text-sm font-semibold flex items-center justify-center gap-2 hover:shadow-lg hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                          >
                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M9.5 2L8 6H4l3.5 3L6 13l3.5-2L13 13l-1.5-4L15 6h-4l-1.5-4zm9 6l-.9 2.3L15 11l2.6 1.7L17 15l.9-2.3L20.5 11l-2.6-1.7zM18.5 17l-.6 1.5L16.5 19l1.4.9.6 1.6.6-1.6 1.4-.9-1.4-.9z"/>
                            </svg>
                            Suggest
                          </button>

                          <label
                            htmlFor={'fileUpload-' + day}
                            className="bg-gradient-to-br from-orange-500 to-orange-600 text-white px-4 py-2.5 rounded-2xl text-sm font-semibold cursor-pointer flex items-center justify-center gap-2 hover:shadow-lg hover:scale-105 transition-all shadow-md"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                            </svg>
                            AI Detect
                          </label>
                          <input
                            type="file"
                            id={'fileUpload-' + day}
                            multiple
                            accept="image/*"
                            onChange={(e) => {
                              setCurrentDay(day);
                              handleFileUpload(e);
                            }}
                            className="hidden"
                          />
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
                                  className={'rounded-3xl overflow-hidden transition-all duration-300 cursor-move ' + (
                                    selectedLocation?.id === location.id
                                      ? 'bg-orange-500 bg-opacity-20 border-2 border-orange-400 shadow-[0_8px_30px_rgb(249,115,22,0.4)] scale-[1.02]'
                                      : 'bg-zinc-800 bg-opacity-60 border border-zinc-700 border-opacity-40 hover:bg-opacity-70 hover:shadow-[0_8px_30px_rgb(0,0,0,0.4)] hover:scale-[1.01]'
                                  )}
                                >
                                  {location.image && (
                                    <div
                                      className="relative h-40 w-full cursor-pointer"
                                      onClick={() => setSelectedLocation(location)}
                                    >
                                      <img
                                        src={location.image}
                                        alt={location.name}
                                        className="absolute inset-0 w-full h-full object-cover"
                                      />
                                      <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent"></div>
                                      <div className="absolute top-3 left-3 w-8 h-8 bg-gradient-to-br from-orange-500 to-orange-600 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-xl border-2 border-white">
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
                                      {location.confidence && (
                                        <div className="absolute top-2 right-11 bg-black bg-opacity-60 backdrop-blur-sm px-2 py-1 rounded-full flex items-center gap-1 text-xs text-white">
                                          <span className="text-yellow-400">OK</span>
                                          {(location.confidence * 100).toFixed(0)}%
                                        </div>
                                      )}
                                      <h3 className="absolute bottom-2 left-2 right-2 text-white text-sm font-semibold truncate drop-shadow-lg">
                                        {location.name}
                                      </h3>
                                    </div>
                                  )}

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
                                    
                                    <button
                                      onClick={() => {
                                        if (showMoreInfo === location.id) {
                                          setShowMoreInfo(null);
                                        } else {
                                          setShowMoreInfo(location.id);
                                          fetchPlaceDetails(location.id, location.name, location.lat, location.lng);
                                        }
                                      }}
                                      className="w-full mt-2 px-3 py-2 bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white text-sm font-medium rounded-xl transition-all hover:scale-105 hover:shadow-lg flex items-center justify-center gap-2"
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
                                    
                                    {/* Expandable Details Section */}
                                    {showMoreInfo === location.id && (
                                      <div className="mt-3 overflow-hidden animate-slideDown">
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
                                      const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(input) + countryParam);
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
  );
}

"use client";

import { useEffect, useState, useRef } from 'react';
import { APIProvider, Map, AdvancedMarker, InfoWindow } from '@vis.gl/react-google-maps';
import Image from 'next/image';

// Add styles to remove Google Maps InfoWindow default styling
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.textContent = `
    .gm-style .gm-style-iw-c {
      background: transparent !important;
      box-shadow: none !important;
      padding: 0 !important;
      border-radius: 0 !important;
    }
    .gm-style .gm-style-iw-d {
      overflow: visible !important;
      background: transparent !important;
    }
    .gm-style .gm-style-iw-t::after {
      display: none !important;
      background: transparent !important;
    }
    .gm-style .gm-style-iw-tc::after {
      display: none !important;
    }
    .gm-style-iw {
      background: transparent !important;
    }
    .gm-style-iw-ch {
      display: none !important;
    }
    .gm-style-iw-tc {
      display: none !important;
    }
  `;
  if (!document.head.querySelector('style[data-map-styles]')) {
    style.setAttribute('data-map-styles', 'true');
    document.head.appendChild(style);
  }
}

interface Location {
  id: string;
  name: string;
  lat: number;
  lng: number;
  image?: string;
  confidence?: number;
  day: number;
  notes?: string;
}

interface Recommendation {
  name: string;
  lat: number;
  lng: number;
  confidence: number;
  image?: string;
}

interface MapComponentProps {
  landmarks: Location[];
  selectedLandmark: Location | null;
  onAddToItinerary?: (name: string, lat: number, lng: number, image?: string) => void;
  recommendations?: Recommendation[];
  onClearRecommendations?: () => void;
  onRemoveRecommendation?: (name: string) => void;
}

// 50 distinct colors for different days
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

const NumberedMarker = ({ number, color }: { number: number; color: string }) => (
  <div style={{
    backgroundColor: color,
    width: '32px',
    height: '32px',
    borderRadius: '50% 50% 50% 0',
    transform: 'rotate(-45deg)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: '3px solid white',
    boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
  }}>
    <span style={{
      transform: 'rotate(45deg)',
      color: 'white',
      fontWeight: 'bold',
      fontSize: '13px',
    }}>
      {number > 9 ? '9+' : number}
    </span>
  </div>
);

const AIRecommendationMarker = ({ rank, isSelected, image, onRemove }: { rank: number; isSelected: boolean; image?: string; onRemove?: () => void }) => (
  <div style={{
    position: 'relative',
    width: '56px',
    height: '56px',
    animation: 'bounce-in 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55)',
    animationDelay: `${rank * 0.1}s`,
    animationFillMode: 'both'
  }}>
    <style jsx>{`
      @keyframes bounce-in {
        0% { transform: scale(0) translateY(-20px); opacity: 0; }
        50% { transform: scale(1.2); }
        100% { transform: scale(1) translateY(0); opacity: 1; }
      }
      @keyframes pulse-ring {
        0% { transform: scale(0.8); opacity: 1; }
        100% { transform: scale(1.8); opacity: 0; }
      }
    `}</style>
    
    {/* Pulsing ring animation */}
    <div style={{
      position: 'absolute',
      inset: '-6px',
      borderRadius: '50%',
      background: 'linear-gradient(135deg, #f97316, #fb923c)',
      animation: 'pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      opacity: 0.6
    }}></div>
    
    {/* X button */}
    {onRemove && (
      <button
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        style={{
          position: 'absolute',
          top: '-8px',
          right: '-8px',
          width: '22px',
          height: '22px',
          borderRadius: '50%',
          background: 'rgba(24, 24, 27, 0.95)',
          border: '2px solid #f97316',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          zIndex: 10,
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          transition: 'all 0.2s ease',
          backdropFilter: 'blur(8px)'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'scale(1.15)';
          e.currentTarget.style.background = '#f97316';
          e.currentTarget.style.borderColor = '#fb923c';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
          e.currentTarget.style.background = 'rgba(24, 24, 27, 0.95)';
          e.currentTarget.style.borderColor = '#f97316';
        }}
      >
        <svg
          style={{
            width: '13px',
            height: '13px',
            color: '#f97316'
          }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    )}
    
    {/* Main marker */}
    <div style={{
      position: 'relative',
      width: '100%',
      height: '100%',
      borderRadius: '50%',
      border: '3px solid',
      borderColor: rank === 1 ? '#f97316' : rank === 2 ? '#fb923c' : '#fbbf24',
      boxShadow: isSelected 
        ? '0 8px 32px rgba(249, 115, 22, 0.6), 0 0 0 2px rgba(249, 115, 22, 0.2)'
        : '0 4px 16px rgba(0, 0, 0, 0.5)',
      cursor: 'pointer',
      transform: isSelected ? 'scale(1.15)' : 'scale(1)',
      transition: 'all 0.3s ease',
      overflow: 'hidden'
    }}>
      {image ? (
        <>
          <img 
            src={image} 
            alt="Location"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />
          {/* Number overlay */}
          <div style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            background: 'linear-gradient(to top, rgba(0,0,0,0.8), transparent)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '4px'
          }}>
            <span style={{
              color: 'white',
              fontWeight: 'bold',
              fontSize: '16px',
              textShadow: '0 2px 4px rgba(0,0,0,0.5)'
            }}>
              {rank}
            </span>
          </div>
        </>
      ) : (
        <div style={{
          width: '100%',
          height: '100%',
          background: isSelected 
            ? 'linear-gradient(135deg, #f97316, #fb923c)'
            : 'linear-gradient(135deg, #18181b, #27272a)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{
            color: 'white',
            fontWeight: 'bold',
            fontSize: '20px',
            textShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            {rank}
          </span>
        </div>
      )}
    </div>
  </div>
);

export default function MapComponent({ landmarks, selectedLandmark, onAddToItinerary, recommendations = [], onClearRecommendations, onRemoveRecommendation }: MapComponentProps) {
  const center = landmarks.length > 0 
    ? { lat: landmarks[0].lat, lng: landmarks[0].lng }
    : { lat: 35.6762, lng: 139.6503 };

  const [openInfoWindowId, setOpenInfoWindowId] = useState<string | null>(null);
  const [showMoreInfo, setShowMoreInfo] = useState<string | null>(null);
  const [placeDetails, setPlaceDetails] = useState<{ [key: string]: any }>({});
  const [loadingDetails, setLoadingDetails] = useState<string | null>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const polylinesRef = useRef<google.maps.Polyline[]>([]);
  
  // State for clicked location (not in itinerary)
  const [clickedPlace, setClickedPlace] = useState<{ lat: number; lng: number; name: string; id: string } | null>(null);
  const [loadingClickedPlace, setLoadingClickedPlace] = useState(false);
  const [modalClickedPlace, setModalClickedPlace] = useState<{ lat: number; lng: number; name: string; id: string } | null>(null);
  const [selectedPhotoIndex, setSelectedPhotoIndex] = useState<number>(0);
  const [currentPhotos, setCurrentPhotos] = useState<string[]>([]);
  
  // State for recommendations
  const [selectedRecommendation, setSelectedRecommendation] = useState<number | null>(null);

  const locationsByDay: { [key: number]: Location[] } = {};
  landmarks.forEach(loc => {
    if (!locationsByDay[loc.day]) {
      locationsByDay[loc.day] = [];
    }
    locationsByDay[loc.day].push(loc);
  });

  // Fetch place details from Google Places API
  const fetchPlaceDetails = async (locationId: string, locationName: string, lat: number, lng: number) => {
    if (placeDetails[locationId]) return;
    
    setLoadingDetails(locationId);
    try {
      const response = await fetch(`/api/place-details?name=${encodeURIComponent(locationName)}&lat=${lat}&lng=${lng}`);
      const data = await response.json();
      setPlaceDetails(prev => ({ ...prev, [locationId]: data }));
    } catch (error) {
      setPlaceDetails(prev => ({ ...prev, [locationId]: null }));
    } finally {
      setLoadingDetails(null);
    }
  };



  // Draw routes when map is loaded
  useEffect(() => {
    if (!map) return;

    // Clear existing polylines
    polylinesRef.current.forEach(p => p.setMap(null));
    polylinesRef.current = [];

    // Fetch and draw routes for each day
    Object.entries(locationsByDay).forEach(async ([day, locs]) => {
      if (locs.length < 2) return;

      try {
        const coordinates = locs.map(loc => `${loc.lng},${loc.lat}`).join(';');
        const response = await fetch(
          `https://router.project-osrm.org/route/v1/driving/${coordinates}?overview=full&geometries=geojson`
        );
        
        const data = await response.json();
        
        if (data.code === 'Ok' && data.routes && data.routes[0]) {
          const routeCoordinates = data.routes[0].geometry.coordinates.map(
            (coord: [number, number]) => ({ lat: coord[1], lng: coord[0] })
          );

          const dayNum = parseInt(day);
          const color = dayColors[dayNum % dayColors.length];

          const polyline = new google.maps.Polyline({
            path: routeCoordinates,
            geodesic: true,
            strokeColor: color,
            strokeOpacity: 0.85,
            strokeWeight: 8,
            map: map,
          });

          polylinesRef.current.push(polyline);
        }
      } catch (error) {
        console.error(`Error fetching route for day ${day}:`, error);
      }
    });

    return () => {
      polylinesRef.current.forEach(p => p.setMap(null));
      polylinesRef.current = [];
    };
  }, [map, landmarks]);

  // Pan to selected landmark
  useEffect(() => {
    if (selectedLandmark && map) {
      map.panTo({ lat: selectedLandmark.lat, lng: selectedLandmark.lng });
      map.setZoom(13);
      setOpenInfoWindowId(selectedLandmark.id);
    }
  }, [selectedLandmark, map]);

  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';

  // Handle POI (Place of Interest) clicks
  useEffect(() => {
    if (!map) return;
    
    const listener = map.addListener('click', async (e: any) => {
      // Only handle clicks on POIs (places with names on the map)
      if (e.placeId) {
        e.stop(); // Prevent default Google info window
        
        const lat = e.latLng.lat();
        const lng = e.latLng.lng();
        
        setLoadingClickedPlace(true);
        setClickedPlace(null);
        setOpenInfoWindowId(null);
        
        try {
          // First get the place name using nearby-place API
          const nearbyResponse = await fetch(`/api/nearby-place?lat=${lat}&lng=${lng}`);
          const nearbyData = await nearbyResponse.json();
          
          if (nearbyData && nearbyData.name && !nearbyData.error) {
            const placeId = `clicked-${Date.now()}`;
            
            // Then get full details using place-details API (which uses new Places API)
            const detailsResponse = await fetch(
              `/api/place-details?name=${encodeURIComponent(nearbyData.name)}&lat=${lat}&lng=${lng}`
            );
            const detailsData = await detailsResponse.json();
            
            setClickedPlace({
              lat: nearbyData.lat || lat,
              lng: nearbyData.lng || lng,
              name: nearbyData.name,
              id: placeId,
            });
            setPlaceDetails(prev => ({ ...prev, [placeId]: detailsData }));
            setOpenInfoWindowId(placeId);
            setShowMoreInfo(placeId);
          }
        } catch (error) {
          console.error('Error fetching place details:', error);
        } finally {
          setLoadingClickedPlace(false);
        }
      }
    });
    
    return () => {
      if (listener) {
        google.maps.event.removeListener(listener);
      }
    };
  }, [map]);

  return (
    <APIProvider apiKey={apiKey} language="en">
      <Map
        defaultCenter={center}
        defaultZoom={12}
        minZoom={3}
        restriction={{
          latLngBounds: {
            north: 85,
            south: -85,
            west: -180,
            east: 180,
          },
          strictBounds: true,
        }}
        mapId="itinerary-map"
        gestureHandling="greedy"
        disableDefaultUI={true}
        zoomControl={true}
        zoomControlOptions={{ position: 7 }}
        fullscreenControl={true}
        fullscreenControlOptions={{ position: 3 }}
        mapTypeControl={true}
        mapTypeControlOptions={{ position: 3 }}
        streetViewControl={true}
        streetViewControlOptions={{ position: 7 }}
        clickableIcons={true}
        onClick={(e: any) => {
          // Handle regular map clicks (close popups)
          if (!e.detail?.placeId) {
            setOpenInfoWindowId(null);
            setClickedPlace(null);
          }
        }}
        onCameraChanged={(ev) => {
          if (ev.map) setMap(ev.map);
        }}
        style={{ width: '100%', height: '100%' }}
      >
        {landmarks.map((location) => {
          const dayLocations = locationsByDay[location.day];
          const dayIndex = dayLocations.findIndex(loc => loc.id === location.id);
          const color = dayColors[location.day % dayColors.length];

          return (
            <div key={location.id}>
              <AdvancedMarker
                position={{ lat: location.lat, lng: location.lng }}
                onClick={() => setOpenInfoWindowId(location.id)}
              >
                <NumberedMarker number={dayIndex + 1} color={color} />
              </AdvancedMarker>

              {openInfoWindowId === location.id && (
                <InfoWindow
                  position={{ lat: location.lat, lng: location.lng }}
                  onCloseClick={() => {
                    setOpenInfoWindowId(null);
                    setShowMoreInfo(null);
                  }}
                  headerDisabled
                >
                  <div style={{
                    position: 'relative',
                    backgroundColor: '#18181b',
                    borderRadius: '16px',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
                    overflow: 'hidden',
                    minWidth: '220px',
                    maxWidth: '280px'
                  }}>
                    {/* Arrow pointer */}
                    <div style={{
                      position: 'absolute',
                      bottom: '-8px',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      width: 0,
                      height: 0,
                      borderLeft: '8px solid transparent',
                      borderRight: '8px solid transparent',
                      borderTop: '8px solid #000000',
                      zIndex: 1000
                    }}></div>
                    
                    {/* Close button */}
                    <button
                      onClick={() => {
                        setOpenInfoWindowId(null);
                        setShowMoreInfo(null);
                      }}
                      className="absolute top-2 right-2 z-10 bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full p-1 transition-all"
                      style={{ width: '24px', height: '24px' }}
                    >
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                    
                    <div className="flex items-center justify-between px-4 py-2" style={{ backgroundColor: color }}>
                      <span className="text-white font-bold text-xs rounded-full px-2 py-1 shadow" style={{ backgroundColor: 'rgba(0,0,0,0.18)' }}>
                        Day {location.day}
                      </span>
                      <span className="text-white text-xs font-semibold">#{dayIndex + 1}</span>
                    </div>
                    {location.image && (
                      <img 
                        src={location.image} 
                        alt={location.name}
                        className="w-full h-32 object-cover"
                      />
                    )}
                    <div className="px-4 py-3">
                      <h3 className="font-bold text-base text-white mb-2">{location.name}</h3>
                      
                      <div className="flex items-center gap-2 mb-2">
                        <span className="flex items-center gap-0.5">
                          {[...Array(5)].map((_, i) => (
                            <svg key={i} className={`w-4 h-4 ${i < 4 ? 'text-yellow-400' : 'text-gray-600'}`} fill="currentColor" viewBox="0 0 20 20">
                              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                          ))}
                        </span>
                        <span className="text-yellow-400 text-xs font-semibold">4.0</span>
                      </div>
                      
                      {location.confidence && (
                        <div className="flex items-center gap-2 mb-2">
                          <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          <span className="text-green-400 text-xs font-semibold">Confidence: {(location.confidence * 100).toFixed(0)}%</span>
                        </div>
                      )}
                      
                      {location.notes && (
                        <div className="mt-2 p-2 bg-zinc-800 bg-opacity-60 rounded-lg border border-zinc-700">
                          <span className="text-stone-400 text-xs italic">{location.notes}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </InfoWindow>
              )}
            </div>
          );
        })}
        
        {/* InfoWindow for clicked places (not in itinerary) */}
        {clickedPlace && openInfoWindowId === clickedPlace.id && (
          <InfoWindow
            position={{ lat: clickedPlace.lat, lng: clickedPlace.lng }}
            onCloseClick={() => {
              setOpenInfoWindowId(null);
              setShowMoreInfo(null);
              setClickedPlace(null);
            }}
            headerDisabled
          >
            <div className="min-w-[220px] max-w-[280px] bg-zinc-900 bg-opacity-95 rounded-2xl shadow-2xl p-0 overflow-hidden relative">
              {/* Maximize button */}
              <button
                onClick={() => {
                  setModalClickedPlace(clickedPlace);
                  fetchPlaceDetails(clickedPlace.id, clickedPlace.name, clickedPlace.lat, clickedPlace.lng);
                }}
                className="absolute top-2 right-[34px] z-10 bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full p-1 transition-all"
                style={{ width: '24px', height: '24px' }}
                title="Maximize"
              >
                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
              </button>
              
              {/* Close button */}
              <button
                onClick={() => {
                  setOpenInfoWindowId(null);
                  setShowMoreInfo(null);
                  setClickedPlace(null);
                }}
                className="absolute top-2 right-2 z-10 bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full p-1 transition-all"
                style={{ width: '24px', height: '24px' }}
              >
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              
              <div className="flex items-center justify-between px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600">
                <span className="text-white font-bold text-xs rounded-full px-2 py-1 shadow" style={{ backgroundColor: 'rgba(0,0,0,0.18)' }}>
                  New Place
                </span>
              </div>
              <div className="px-4 py-3">
                <h3 className="font-bold text-base text-white mb-3">{clickedPlace.name}</h3>
                
                {loadingDetails === clickedPlace.id ? (
                  <div className="py-6 flex items-center justify-center">
                    <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-orange-500"></div>
                  </div>
                ) : placeDetails[clickedPlace.id] ? (
                  <div className="bg-zinc-800 bg-opacity-50 rounded-xl p-3 space-y-2 border border-zinc-700 border-opacity-30 mb-3 max-h-64 overflow-y-auto">
                    {placeDetails[clickedPlace.id].rating && (
                      <div className="flex items-center gap-2">
                        <div className="flex items-center gap-0.5">
                          {[...Array(5)].map((_, i) => (
                            <svg
                              key={i}
                              className={`w-4 h-4 ${i < Math.floor(placeDetails[clickedPlace.id].rating) ? 'text-yellow-400' : 'text-gray-600'}`}
                              fill="currentColor"
                              viewBox="0 0 20 20"
                            >
                              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                          ))}
                        </div>
                        <span className="text-white font-semibold text-sm">{placeDetails[clickedPlace.id].rating.toFixed(1)}</span>
                        {placeDetails[clickedPlace.id].totalRatings && (
                          <span className="text-stone-400 text-xs">({placeDetails[clickedPlace.id].totalRatings.toLocaleString()})</span>
                        )}
                      </div>
                    )}
                    
                    {placeDetails[clickedPlace.id].photos && placeDetails[clickedPlace.id].photos.length > 0 && (
                      <div className="flex gap-1 overflow-x-auto">
                        {placeDetails[clickedPlace.id].photos.slice(0, 3).map((photoUrl: string, idx: number) => (
                          <div key={idx} className="flex-shrink-0 w-20 h-16 rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700 border-opacity-30">
                            <Image
                              src={photoUrl}
                              alt={`${clickedPlace.name} photo ${idx + 1}`}
                              width={80}
                              height={64}
                              className="w-full h-full object-cover"
                            />
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {placeDetails[clickedPlace.id].description && (
                      <p className="text-stone-300 text-xs leading-relaxed">{placeDetails[clickedPlace.id].description.slice(0, 150)}...</p>
                    )}
                    
                    {placeDetails[clickedPlace.id].openingHours && (
                      <div className="flex items-start gap-1.5">
                        <svg className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="text-stone-300 text-xs">{placeDetails[clickedPlace.id].openingHours}</p>
                      </div>
                    )}
                    
                    {placeDetails[clickedPlace.id].address && (
                      <div className="flex items-start gap-1.5">
                        <svg className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <p className="text-stone-300 text-xs">{placeDetails[clickedPlace.id].address}</p>
                      </div>
                    )}
                    
                    {placeDetails[clickedPlace.id].website && (
                      <a
                        href={placeDetails[clickedPlace.id].website}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1.5 text-orange-400 hover:text-orange-300 text-xs transition-colors"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                        Visit Website
                      </a>
                    )}
                  </div>
                ) : null}
                
                {onAddToItinerary && (
                  <button
                    onClick={() => {
                      onAddToItinerary(clickedPlace.name, clickedPlace.lat, clickedPlace.lng);
                      setOpenInfoWindowId(null);
                      setShowMoreInfo(null);
                      setClickedPlace(null);
                    }}
                    className="w-full px-3 py-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white text-xs font-medium rounded-xl transition-all hover:scale-105 flex items-center justify-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    Add to Itinerary
                  </button>
                )}
              </div>
            </div>
          </InfoWindow>
        )}
        
        {loadingClickedPlace && (
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-zinc-900 bg-opacity-90 rounded-xl p-4 shadow-xl">
            <div className="flex items-center gap-3">
              <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-orange-500"></div>
              <span className="text-white text-sm">Finding place...</span>
            </div>
          </div>
        )}

        {/* AI Recommendation Markers */}
        {recommendations.map((rec, index) => (
          <div key={`rec-${index}`}>
            <AdvancedMarker
              position={{ lat: rec.lat, lng: rec.lng }}
              onClick={() => setSelectedRecommendation(index)}
            >
              <AIRecommendationMarker 
                rank={index + 1} 
                isSelected={selectedRecommendation === index} 
                image={rec.image}
                onRemove={onRemoveRecommendation ? () => onRemoveRecommendation(rec.name) : undefined}
              />
            </AdvancedMarker>

            {selectedRecommendation === index && (
              <InfoWindow
                position={{ lat: rec.lat, lng: rec.lng }}
                onCloseClick={() => setSelectedRecommendation(null)}
                headerDisabled
              >
                <div style={{
                  position: 'relative',
                  background: 'linear-gradient(135deg, #18181b, #27272a)',
                  borderRadius: '16px',
                  boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
                  overflow: 'hidden',
                  minWidth: '240px',
                  maxWidth: '280px',
                  border: '2px solid #f97316'
                }}>
                  <div style={{
                    position: 'absolute',
                    bottom: '-10px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: 0,
                    height: 0,
                    borderLeft: '10px solid transparent',
                    borderRight: '10px solid transparent',
                    borderTop: '10px solid #f97316',
                    zIndex: 1000
                  }}></div>

                  <button
                    onClick={() => setSelectedRecommendation(null)}
                    className="absolute top-2 right-2 z-10 bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full p-1 transition-all"
                    style={{ width: '24px', height: '24px' }}
                  >
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>

                  {rec.image && (
                    <img 
                      src={rec.image} 
                      alt={rec.name}
                      className="w-full h-32 object-cover"
                    />
                  )}

                  <div className="px-4 py-3" style={{
                    background: 'linear-gradient(135deg, #f97316, #fb923c)',
                  }}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-white font-bold text-sm">AI Recommendation</span>
                      <span className="bg-black bg-opacity-30 text-white text-xs font-bold px-2 py-0.5 rounded-full">#{index + 1}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                      </svg>
                      <span className="text-white text-xs font-semibold">Confidence: {(rec.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  <div className="px-4 py-3">
                    <h3 className="font-bold text-base text-white mb-3">{rec.name}</h3>
                    
                    <button
                      onClick={() => {
                        if (onAddToItinerary) {
                          onAddToItinerary(rec.name, rec.lat, rec.lng, rec.image);
                          setSelectedRecommendation(null);
                          // Don't clear recommendations - keep them visible
                        }
                      }}
                      className="w-full bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white font-semibold py-2.5 px-4 rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-xl"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                      </svg>
                      Add to Itinerary
                    </button>
                  </div>
                </div>
              </InfoWindow>
            )}
          </div>
        ))}
      </Map>
      
      {/* Modal for clicked places */}
      {modalClickedPlace && (
        <>
          <div 
            className="fixed inset-0 bg-transparent backdrop-blur-md z-[10000] animate-fadeIn"
            onClick={() => setModalClickedPlace(null)}
          />
          <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4 pointer-events-none">
            <div 
              className="bg-zinc-900 bg-opacity-95 rounded-2xl shadow-2xl w-[70%] max-w-3xl max-h-[75vh] overflow-y-auto pointer-events-auto animate-scaleIn border border-zinc-700 border-opacity-40 relative"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => setModalClickedPlace(null)}
                className="absolute top-4 right-4 z-10 w-10 h-10 bg-zinc-800 bg-opacity-80 hover:bg-opacity-100 backdrop-blur-sm rounded-full flex items-center justify-center transition-all hover:scale-110"
              >
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              
              <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-blue-500 to-blue-600">
                <span className="text-white font-bold text-sm rounded-full px-3 py-1.5 shadow" style={{ backgroundColor: 'rgba(0,0,0,0.18)' }}>
                  New Place
                </span>
              </div>
              
              <div className="px-6 py-5">
                <h2 className="text-white text-2xl font-bold mb-4">{modalClickedPlace.name}</h2>
                
                {loadingDetails === modalClickedPlace.id ? (
                  <div className="py-12 flex items-center justify-center">
                    <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-orange-500"></div>
                  </div>
                ) : placeDetails[modalClickedPlace.id] ? (
                  <div className="space-y-4">
                    {placeDetails[modalClickedPlace.id].rating && (
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-1">
                          {[...Array(5)].map((_, i) => (
                            <svg
                              key={i}
                              className={`w-5 h-5 ${i < Math.floor(placeDetails[modalClickedPlace.id].rating) ? 'text-yellow-400' : 'text-gray-600'}`}
                              fill="currentColor"
                              viewBox="0 0 20 20"
                            >
                              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                          ))}
                        </div>
                        <span className="text-white font-semibold text-lg">{placeDetails[modalClickedPlace.id].rating.toFixed(1)}</span>
                        {placeDetails[modalClickedPlace.id].totalRatings && (
                          <span className="text-stone-400 text-sm">({placeDetails[modalClickedPlace.id].totalRatings.toLocaleString()} reviews)</span>
                        )}
                      </div>
                    )}
                    
                    {placeDetails[modalClickedPlace.id].photos && placeDetails[modalClickedPlace.id].photos.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="text-white font-semibold text-sm flex items-center gap-2">
                          <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                          Photos
                        </h4>
                        <div className="grid grid-cols-3 gap-2">
                          {placeDetails[modalClickedPlace.id].photos.slice(0, 6).map((photoUrl: string, idx: number) => (
                            <div 
                              key={idx} 
                              className="aspect-video rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700 border-opacity-30 cursor-pointer hover:scale-105 transition-transform"
                              onClick={() => {
                                setCurrentPhotos(placeDetails[modalClickedPlace.id].photos);
                                setSelectedPhotoIndex(idx);
                              }}
                            >
                              <Image
                                src={photoUrl}
                                alt={`${modalClickedPlace.name} photo ${idx + 1}`}
                                width={200}
                                height={150}
                                className="w-full h-full object-cover"
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {placeDetails[modalClickedPlace.id].description && (
                      <div className="bg-zinc-800 bg-opacity-50 rounded-xl p-4 border border-zinc-700 border-opacity-30">
                        <p className="text-stone-300 text-sm leading-relaxed">{placeDetails[modalClickedPlace.id].description}</p>
                      </div>
                    )}
                    
                    <div className="grid grid-cols-1 gap-3">
                      {placeDetails[modalClickedPlace.id].openingHours && (
                        <div className="flex items-start gap-2">
                          <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <p className="text-stone-300 text-sm">{placeDetails[modalClickedPlace.id].openingHours}</p>
                        </div>
                      )}
                      
                      {placeDetails[modalClickedPlace.id].phone && (
                        <div className="flex items-center gap-2">
                          <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                          </svg>
                          <p className="text-stone-300 text-sm">{placeDetails[modalClickedPlace.id].phone}</p>
                        </div>
                      )}
                      
                      {placeDetails[modalClickedPlace.id].address && (
                        <div className="flex items-start gap-2">
                          <svg className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                          <p className="text-stone-300 text-sm">{placeDetails[modalClickedPlace.id].address}</p>
                        </div>
                      )}
                      
                      {placeDetails[modalClickedPlace.id].website && (
                        <a
                          href={placeDetails[modalClickedPlace.id].website}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-2 text-orange-400 hover:text-orange-300 text-sm transition-colors"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                          Visit Website
                        </a>
                      )}
                    </div>
                    
                    {onAddToItinerary && (
                      <button
                        onClick={() => {
                          onAddToItinerary(modalClickedPlace.name, modalClickedPlace.lat, modalClickedPlace.lng);
                          setModalClickedPlace(null);
                          setClickedPlace(null);
                          setOpenInfoWindowId(null);
                        }}
                        className="w-full px-4 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white text-sm font-medium rounded-xl transition-all hover:scale-105 flex items-center justify-center gap-2 mt-4"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        Add to Itinerary
                      </button>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
          
          {/* Photo viewer modal */}
          {currentPhotos.length > 0 && (
            <div 
              className="fixed inset-0 z-[20000] flex items-center justify-center bg-black bg-opacity-90 backdrop-blur-sm" 
              onClick={() => setCurrentPhotos([])}
            >
              <div className="relative max-w-7xl max-h-[90vh] w-full px-4">
                {/* Close Button */}
                <button
                  onClick={() => setCurrentPhotos([])}
                  className="absolute top-4 right-4 w-12 h-12 bg-zinc-800 bg-opacity-80 hover:bg-opacity-100 backdrop-blur-sm rounded-full flex items-center justify-center transition-all hover:scale-110 z-10"
                >
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
                
                {/* Left Arrow */}
                {currentPhotos.length > 1 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedPhotoIndex((selectedPhotoIndex - 1 + currentPhotos.length) % currentPhotos.length);
                    }}
                    className="absolute left-4 top-1/2 -translate-y-1/2 w-12 h-12 bg-zinc-800 bg-opacity-80 hover:bg-opacity-100 backdrop-blur-sm rounded-full flex items-center justify-center transition-all hover:scale-110 z-10"
                  >
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                  </button>
                )}
                
                {/* Right Arrow */}
                {currentPhotos.length > 1 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedPhotoIndex((selectedPhotoIndex + 1) % currentPhotos.length);
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
        </>
      )}
    </APIProvider>
  );
}

"use client";

import { useEffect, useState, useRef } from 'react';
import { APIProvider, Map, AdvancedMarker, InfoWindow } from '@vis.gl/react-google-maps';

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

interface MapComponentProps {
  landmarks: Location[];
  selectedLandmark: Location | null;
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

export default function MapComponent({ landmarks, selectedLandmark }: MapComponentProps) {
  const center = landmarks.length > 0 
    ? { lat: landmarks[0].lat, lng: landmarks[0].lng }
    : { lat: 35.6762, lng: 139.6503 };

  const [openInfoWindowId, setOpenInfoWindowId] = useState<string | null>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const polylinesRef = useRef<google.maps.Polyline[]>([]);

  const locationsByDay: { [key: number]: Location[] } = {};
  landmarks.forEach(loc => {
    if (!locationsByDay[loc.day]) {
      locationsByDay[loc.day] = [];
    }
    locationsByDay[loc.day].push(loc);
  });

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
            strokeOpacity: 0.6,
            strokeWeight: 3,
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

  return (
    <APIProvider apiKey={apiKey}>
      <Map
        defaultCenter={center}
        defaultZoom={12}
        mapId="itinerary-map"
        gestureHandling="greedy"
        disableDefaultUI={false}
        zoomControl={true}
        onCameraChanged={(ev) => {
          if (ev.map && !map) setMap(ev.map);
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
                  onCloseClick={() => setOpenInfoWindowId(null)}
                  headerDisabled
                >
                  <div className="min-w-[220px] max-w-[280px] bg-zinc-900 bg-opacity-95 rounded-2xl shadow-2xl p-0 overflow-hidden">
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
                      <h3 className="font-bold text-base text-white mb-1">{location.name}</h3>
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
                      <div className="flex items-center gap-2 mb-2">
                        <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                        </svg>
                        <span className="text-stone-300 text-xs">Popular spot</span>
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
      </Map>
    </APIProvider>
  );
}

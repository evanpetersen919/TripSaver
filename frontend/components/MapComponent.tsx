"use client";

import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, ZoomControl } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

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

const createNumberedIcon = (number: number, color: string) => {
  return L.divIcon({
    className: 'custom-div-icon',
    html: `<div style="
      background-color: ${color};
      width: 28px;
      height: 28px;
      border-radius: 50% 50% 50% 0;
      transform: rotate(-45deg);
      display: flex;
      align-items: center;
      justify-content: center;
      border: 2px solid white;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    ">
      <span style="
        transform: rotate(45deg);
        color: white;
        font-weight: bold;
        font-size: 12px;
      ">${number > 9 ? '9+' : number}</span>
    </div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 28],
  });
};

function MapController({ selectedLandmark }: { selectedLandmark: Location | null }) {
  const map = useMap();
  
  useEffect(() => {
    if (selectedLandmark) {
      map.flyTo([selectedLandmark.lat, selectedLandmark.lng], 13, {
        duration: 1.5
      });
    }
  }, [selectedLandmark, map]);
  
  return null;
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

export default function MapComponent({ landmarks, selectedLandmark }: MapComponentProps) {
  const center: [number, number] = landmarks.length > 0 
    ? [landmarks[0].lat, landmarks[0].lng]
    : [35.6762, 139.6503];

  const [routesByDay, setRoutesByDay] = useState<{ [key: string]: [number, number][] }>({});

  const locationsByDay: { [key: number]: Location[] } = {};
  landmarks.forEach(loc => {
    if (!locationsByDay[loc.day]) {
      locationsByDay[loc.day] = [];
    }
    locationsByDay[loc.day].push(loc);
  });

  // Fetch road-following routes for each day
  useEffect(() => {
    const fetchRoutes = async () => {
      const newRoutes: { [key: string]: [number, number][] } = {};

      for (const [day, locs] of Object.entries(locationsByDay)) {
        if (locs.length < 2) continue;

        try {
          // Use OSRM (free routing service) - no API key needed
          const coordinates = locs.map(loc => `${loc.lng},${loc.lat}`).join(';');
          const response = await fetch(
            `https://router.project-osrm.org/route/v1/driving/${coordinates}?overview=full&geometries=geojson`
          );
          
          const data = await response.json();
          
          if (data.code === 'Ok' && data.routes && data.routes[0]) {
            // Convert GeoJSON coordinates to Leaflet format [lat, lng]
            const routeCoordinates: [number, number][] = data.routes[0].geometry.coordinates.map(
              (coord: [number, number]) => [coord[1], coord[0]] // GeoJSON is [lng, lat], Leaflet needs [lat, lng]
            );
            newRoutes[`day-${day}`] = routeCoordinates;
          }
        } catch (error) {
          console.error(`Error fetching route for day ${day}:`, error);
          // Fallback to straight lines if routing fails
          newRoutes[`day-${day}`] = locs.map(loc => [loc.lat, loc.lng]);
        }
      }

      setRoutesByDay(newRoutes);
    };

    if (Object.keys(locationsByDay).length > 0) {
      fetchRoutes();
    }
  }, [landmarks]);

  return (
    <MapContainer
      center={center}
      zoom={12}
      minZoom={3}
      maxZoom={18}
      style={{ height: '100%', width: '100%' }}
      zoomControl={false}
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?lang=en"
      />
      
      <MapController selectedLandmark={selectedLandmark} />
      <ZoomControl position="topright" />
      
      {landmarks.map((location, index) => {
        const dayLocations = locationsByDay[location.day];
        const dayIndex = dayLocations.findIndex(loc => loc.id === location.id);
        const color = dayColors[location.day % dayColors.length];
        
        return (
          <Marker
            key={location.id}
            position={[location.lat, location.lng]}
            icon={createNumberedIcon(dayIndex + 1, color)}
          >
            <Popup>
              <div className="min-w-[200px]">
                {location.image && (
                  <img 
                    src={location.image} 
                    alt={location.name}
                    className="w-full h-32 object-cover rounded mb-2"
                  />
                )}
                <h3 className="font-semibold text-sm mb-1">{location.name}</h3>
                <p className="text-xs text-gray-600">
                  {location.lat.toFixed(4)}, {location.lng.toFixed(4)}
                </p>
                {location.confidence && (
                  <p className="text-xs text-green-600 mt-1">
                    Confidence: {(location.confidence * 100).toFixed(0)}%
                  </p>
                )}
                {location.notes && (
                  <p className="text-xs text-gray-700 mt-2 italic">{location.notes}</p>
                )}
              </div>
            </Popup>
          </Marker>
        );
      })}

      {Object.entries(routesByDay).map(([key, positions]) => {
        if (!positions || positions.length < 2) return null;
        
        // Extract day number from key like "day-1"
        const dayNum = parseInt(key.split('-')[1]);
        const color = dayColors[dayNum % dayColors.length];
        
        return (
          <Polyline
            key={key}
            positions={positions}
            color={color}
            weight={3}
            opacity={0.6}
            dashArray="10, 10"
          />
        );
      })}
    </MapContainer>
  );
}

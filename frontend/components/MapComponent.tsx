"use client";

import { useEffect } from 'react';
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

const createNumberedIcon = (number: number) => {
  return L.divIcon({
    className: 'custom-div-icon',
    html: `<div style="
      background-color: #f97316;
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

export default function MapComponent({ landmarks, selectedLandmark }: MapComponentProps) {
  const center: [number, number] = landmarks.length > 0 
    ? [landmarks[0].lat, landmarks[0].lng]
    : [35.6762, 139.6503];

  const locationsByDay: { [key: number]: Location[] } = {};
  landmarks.forEach(loc => {
    if (!locationsByDay[loc.day]) {
      locationsByDay[loc.day] = [];
    }
    locationsByDay[loc.day].push(loc);
  });

  return (
    <MapContainer
      center={center}
      zoom={12}
      style={{ height: '100%', width: '100%' }}
      zoomControl={false}
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
      />
      
      <MapController selectedLandmark={selectedLandmark} />
      <ZoomControl position="topright" />
      
      {landmarks.map((location, index) => {
        const dayLocations = locationsByDay[location.day];
        const dayIndex = dayLocations.findIndex(loc => loc.id === location.id);
        
        return (
          <Marker
            key={location.id}
            position={[location.lat, location.lng]}
            icon={createNumberedIcon(dayIndex + 1)}
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

      {Object.entries(locationsByDay).map(([day, locs]) => {
        if (locs.length < 2) return null;
        
        const positions: [number, number][] = locs.map(loc => [loc.lat, loc.lng]);
        
        return (
          <Polyline
            key={'day-' + day}
            positions={positions}
            color="#f97316"
            weight={3}
            opacity={0.6}
            dashArray="10, 10"
          />
        );
      })}
    </MapContainer>
  );
}

"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import { useParams } from "next/navigation";

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

interface ItineraryData {
  id: string;
  tripName: string;
  destination: string;
  destinationImage: string;
  destinationLat: number;
  destinationLng: number;
  startDate: string;
  endDate: string;
  locations: Location[];
  daySubheadings: { [key: number]: string };
  timestamp: string;
}

// 50 distinct colors for different days (matches dashboard)
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

export default function SharedItinerary() {
  const params = useParams();
  const [itinerary, setItinerary] = useState<ItineraryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null);

  useEffect(() => {
    const loadItinerary = () => {
      try {
        const shareId = params.id as string;
        const storedData = localStorage.getItem(`itinerary-${shareId}`);
        
        if (!storedData) {
          setError(true);
          setLoading(false);
          return;
        }

        const data = JSON.parse(storedData) as ItineraryData;
        setItinerary(data);
        setLoading(false);
      } catch (err) {
        console.error('Error loading itinerary:', err);
        setError(true);
        setLoading(false);
      }
    };

    loadItinerary();
  }, [params.id]);

  const getTotalDays = () => {
    if (!itinerary?.startDate || !itinerary?.endDate) return 1;
    const start = new Date(itinerary.startDate);
    const end = new Date(itinerary.endDate);
    const days = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)) + 1;
    return days > 0 ? days : 1;
  };

  const getLocationsByDay = (day: number) => {
    if (!itinerary) return [];
    return itinerary.locations.filter(loc => loc.day === day);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500 mx-auto mb-4"></div>
          <p className="text-stone-400">Loading itinerary...</p>
        </div>
      </div>
    );
  }

  if (error || !itinerary) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="text-center max-w-md px-4">
          <svg className="w-16 h-16 text-stone-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h1 className="text-2xl font-bold text-white mb-2">Itinerary Not Found</h1>
          <p className="text-stone-400 mb-6">This itinerary link may be invalid or expired.</p>
          <a 
            href="/" 
            className="inline-block px-6 py-3 bg-orange-600 hover:bg-orange-500 text-white rounded-lg transition-colors"
          >
            Go to Home
          </a>
        </div>
      </div>
    );
  }

  const totalDays = getTotalDays();

  return (
    <div className="h-screen flex flex-col bg-zinc-950">
      {/* Header */}
      <div className="h-16 bg-zinc-900 backdrop-blur-3xl bg-opacity-95 border-b border-zinc-800 border-opacity-50 flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-4">
          <a href="/" className="text-orange-400 hover:text-orange-300 transition-colors">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </a>
          <div>
            <h1 className="text-xl font-bold text-white">{itinerary.tripName}</h1>
            {itinerary.startDate && itinerary.endDate && (
              <p className="text-xs text-stone-400">{itinerary.startDate} - {itinerary.endDate}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="px-3 py-1.5 bg-zinc-800 rounded-lg">
            <span className="text-xs text-stone-400">Read-Only View</span>
          </div>
        </div>
      </div>

      <div className="flex-1 relative">
        <div className="absolute inset-0">
          <MapComponent 
            landmarks={itinerary.locations} 
            selectedLandmark={selectedLocation}
            onAddToItinerary={() => {}} // Disabled in read-only mode
          />
        </div>

        {/* Sidebar with Itinerary */}
        <div 
          className="absolute top-5 left-5 bottom-5 w-96 min-w-[384px] bg-zinc-900 bg-opacity-95 backdrop-blur-3xl border border-zinc-800 border-opacity-50 flex flex-col shadow-[0_8px_32px_0_rgba(0,0,0,0.6)] rounded-3xl z-[1000]"
        >
          <div className="flex flex-col h-full">
            {(itinerary.destination || itinerary.tripName) && (
              <div className="relative h-48 border-b border-zinc-800 border-opacity-50 overflow-hidden rounded-t-3xl">
                {itinerary.destinationImage ? (
                  <Image
                    src={itinerary.destinationImage}
                    alt={itinerary.destination}
                    fill
                    className="object-cover"
                    unoptimized
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-orange-600 to-orange-800"></div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-zinc-900 via-zinc-900/50 to-transparent"></div>
                <div className="absolute bottom-4 left-4 right-4">
                  <h2 className="text-2xl font-bold text-white mb-1">{itinerary.destination || itinerary.tripName}</h2>
                  {itinerary.startDate && itinerary.endDate && (
                    <p className="text-sm text-stone-300">{itinerary.startDate} - {itinerary.endDate}</p>
                  )}
                </div>
              </div>
            )}

            {/* Itinerary Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {Array.from({ length: totalDays }, (_, i) => i + 1).map((day) => {
                const dayLocations = getLocationsByDay(day);
                const dayColor = dayColors[(day - 1) % dayColors.length];
                
                return (
                  <div key={day} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white"
                          style={{ backgroundColor: dayColor }}
                        >
                          {day}
                        </div>
                        <h3 className="text-base font-semibold text-white">
                          {itinerary.daySubheadings[day] || `Day ${day}`}
                        </h3>
                      </div>
                    </div>

                    {dayLocations.length === 0 ? (
                      <div className="text-sm text-stone-500 italic pl-10">No locations added</div>
                    ) : (
                      <div className="space-y-3 pl-10">
                        {dayLocations.map((location, index) => (
                          <div
                            key={location.id}
                            onClick={() => setSelectedLocation(location)}
                            className="group bg-zinc-800 bg-opacity-50 rounded-2xl overflow-hidden border border-zinc-700 border-opacity-50 hover:border-opacity-100 hover:border-orange-500 transition-all cursor-pointer"
                          >
                            {location.image && (
                              <div className="relative h-32 w-full overflow-hidden rounded-t-2xl">
                                <Image
                                  src={location.image}
                                  alt={location.name}
                                  fill
                                  className="object-cover group-hover:scale-105 transition-transform duration-300"
                                  unoptimized
                                />
                              </div>
                            )}
                            <div className="p-3">
                              <h4 className="font-medium text-white text-sm mb-1">{location.name}</h4>
                              {location.notes && (
                                <p className="text-xs text-stone-400 mt-2">{location.notes}</p>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

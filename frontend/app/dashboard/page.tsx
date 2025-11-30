"use client";

import React from "react";
import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import { useRouter } from "next/navigation";
import SearchSuggestions from "@/components/SearchSuggestions";

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

    const destinationImages: { [key: string]: string } = {
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
    if (dest) {
      const destLower = dest.toLowerCase();
      if (destinationImages[destLower]) {
        setDestinationImage(destinationImages[destLower]);
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

  const getLocationImage = (locationName: string): string => {
    const timestamp = Date.now();
    return 'https://source.unsplash.com/400x300/?' + encodeURIComponent(locationName) + '&' + timestamp;
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
    
    const imageUrl = getLocationImage(placeName);
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
    
    setTimeout(() => {
      const mockLocations: Location[] = [
        {
          id: Date.now().toString() + '-1',
          name: 'Senso-ji Temple',
          lat: 35.7148,
          lng: 139.7967,
          image: getLocationImage('Senso-ji Temple Tokyo'),
          day: day,
        },
        {
          id: Date.now().toString() + '-2',
          name: 'Meiji Shrine',
          lat: 35.6764,
          lng: 139.6993,
          image: getLocationImage('Meiji Shrine Tokyo'),
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

  const handleSearchChange = async (value: string) => {
    setSearchInput(value);
    if (value.trim().length > 1) {
      try {
        const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(value));
        const data = await response.json();
        const suggestions = data.landmarks.map((l: any) => l.name);
        setSearchSuggestions(suggestions.slice(0, 8));
        setShowSuggestions(suggestions.length !== 0);
      } catch (error) {
        console.error('Error searching landmarks:', error);
        setShowSuggestions(false);
      }
    } else {
      setShowSuggestions(false);
    }
  };

  const handleBottomSearchChange = async (day: number, value: string) => {
    setBottomSearchInput({ ...bottomSearchInput, [day]: value });
    if (value.trim().length > 1) {
      try {
        const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(value));
        const data = await response.json();
        const suggestions = data.landmarks.map((l: any) => l.name);
        setBottomSearchSuggestions({ ...bottomSearchSuggestions, [day]: suggestions.slice(0, 8) });
        setShowBottomSuggestions({ ...showBottomSuggestions, [day]: suggestions.length !== 0 });
      } catch (error) {
        console.error('Error searching landmarks:', error);
        setShowBottomSuggestions({ ...showBottomSuggestions, [day]: false });
      }
    } else {
      setShowBottomSuggestions({ ...showBottomSuggestions, [day]: false });
    }
  };

  const handleInsertSearchChange = async (key: string, value: string) => {
    setInsertSearchInput({ ...insertSearchInput, [key]: value });
    if (value.trim().length > 1) {
      try {
        const response = await fetch('/api/landmarks/search?q=' + encodeURIComponent(value));
        const data = await response.json();
        const suggestions = data.landmarks.map((l: any) => l.name);
        setInsertSearchSuggestions({ ...insertSearchSuggestions, [key]: suggestions.slice(0, 8) });
        setShowInsertSuggestions({ ...showInsertSuggestions, [key]: suggestions.length !== 0 });
      } catch (error) {
        console.error('Error searching landmarks:', error);
        setShowInsertSuggestions({ ...showInsertSuggestions, [key]: false });
      }
    } else {
      setShowInsertSuggestions({ ...showInsertSuggestions, [key]: false });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-900 via-stone-900 to-zinc-900 flex flex-col">
      <div className="bg-zinc-900 bg-opacity-50 backdrop-blur-md border-b border-stone-700 border-opacity-20 px-6 py-3">
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
              className="text-base font-medium text-stone-100 bg-transparent border-none outline-none focus:text-orange-400 transition-colors placeholder-stone-600"
            />
            {startDate && endDate && (
              <>
                <div className="h-5 w-px bg-stone-700 bg-opacity-30"></div>
                <span className="text-sm text-stone-300">{startDate} - {endDate}</span>
              </>
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

        <div className="absolute top-4 left-4 bottom-4 w-96 min-w-[384px] bg-zinc-900 bg-opacity-90 backdrop-blur-xl border border-stone-700 border-opacity-30 flex flex-col shadow-2xl rounded-2xl overflow-hidden z-[1000]">
          {(destination || tripName) && (
            <div className="relative h-40 border-b border-stone-700 border-opacity-30 overflow-hidden">
              {destinationImage ? (
                <div className="absolute inset-0 w-full h-full">
                  <img 
                    src={destinationImage} 
                    alt={destination}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent"></div>
                </div>
              ) : (
                <div className="absolute inset-0 bg-gradient-to-br from-orange-900/20 to-purple-900/20"></div>
              )}
              <div className="absolute bottom-3 left-4 right-4">
                <h1 className="text-xl font-bold text-white mb-0.5 drop-shadow-lg">{tripName}</h1>
                {destination && (
                  <p className="text-sm text-stone-300 drop-shadow">{destination}</p>
                )}
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {Array.from({ length: getTotalDays() }, (_, i) => i + 1).map((day) => {
              const dayLocations = getLocationsByDay(day);
              const isExpanded = currentDay === day;
              
              return (
                <div key={day} className="mb-3 mx-3 first:mt-3">
                  <div className="bg-zinc-900 bg-opacity-60 backdrop-blur-sm rounded-xl border border-stone-700 border-opacity-30 overflow-hidden shadow-lg">
                    <button
                      onClick={() => setCurrentDay(isExpanded ? 0 : day)}
                      className="w-full flex items-center justify-between px-4 py-3 hover:bg-zinc-800 hover:bg-opacity-30 transition-all"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-orange-500 bg-opacity-20 flex items-center justify-center text-orange-400 text-sm font-semibold">
                          {day}
                        </div>
                        <span className="text-white text-base font-medium">Day {day}</span>
                        {dayLocations.length !== 0 && (
                          <span className="text-sm text-stone-400">({dayLocations.length})</span>
                        )}
                      </div>
                      <span
                        className={'text-stone-500 transition-transform inline-block ' + (isExpanded ? 'rotate-180' : '')}
                      >
                        v
                      </span>
                    </button>

                    {isExpanded && (
                      <div className="bg-black bg-opacity-20 px-4 py-3">
                        <input
                          type="text"
                          value={daySubheadings[day] || ''}
                          onChange={(e) => setDaySubheadings({ ...daySubheadings, [day]: e.target.value })}
                          placeholder="Add description for this day..."
                          spellCheck={false}
                          className="w-full px-3 py-2 mb-3 bg-transparent border-b border-stone-700 border-opacity-30 text-white text-sm placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors"
                        />

                        <div className="relative mb-3">
                          <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-stone-500 z-10">@</span>
                          <input
                            type="text"
                            value={searchInput}
                            placeholder="Add a place..."
                            onChange={(e) => handleSearchChange(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && searchInput.trim()) {
                                handleAddPlace(day, searchInput);
                                setSearchInput('');
                                setShowSuggestions(false);
                              }
                            }}
                            onFocus={() => {
                              if (searchInput.trim()) {
                                setShowSuggestions(searchSuggestions.length !== 0);
                              }
                            }}
                            className="w-full pl-10 pr-3 py-2 bg-zinc-800 bg-opacity-40 border border-stone-700 border-opacity-30 rounded-lg text-white text-sm placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors" 
                          />
                          {showSuggestions && searchSuggestions.length > 0 ? (
                            <div className="absolute top-full left-0 right-0 mt-1 z-50">
                              <SearchSuggestions
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
                            className="bg-purple-500 bg-opacity-15 text-white px-3 py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-2 hover:bg-opacity-25 transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-purple-500 border-opacity-30"
                          >
                            <span>*</span>
                            Suggest
                          </button>

                          <label
                            htmlFor={'fileUpload-' + day}
                            className="bg-orange-500 bg-opacity-15 text-white px-3 py-2 rounded-lg text-sm font-medium cursor-pointer flex items-center justify-center gap-2 hover:bg-opacity-25 transition-all border border-orange-500 border-opacity-30"
                          >
                            <span>+</span>
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
                            <p className="text-stone-500 text-sm">No locations yet</p>
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
                                  className={'rounded-lg overflow-hidden transition-all cursor-move ' + (
                                    selectedLocation?.id === location.id
                                      ? 'bg-orange-500 bg-opacity-20 border border-orange-500 border-opacity-40 shadow-lg'
                                      : 'bg-zinc-800 bg-opacity-40 border border-stone-700 border-opacity-20 hover:bg-opacity-50'
                                  )}
                                >
                                  {location.image && (
                                    <div
                                      className="relative h-32 w-full cursor-pointer"
                                      onClick={() => setSelectedLocation(location)}
                                    >
                                      <img
                                        src={location.image}
                                        alt={location.name}
                                        className="absolute inset-0 w-full h-full object-cover"
                                      />
                                      <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent"></div>
                                      <div className="absolute top-2 left-2 w-7 h-7 bg-orange-500 rounded-lg flex items-center justify-center text-white text-sm font-semibold shadow-lg">
                                        {index + 1}
                                      </div>
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          handleDeleteLocation(location.id);
                                        }}
                                        className="absolute top-2 right-2 w-7 h-7 bg-zinc-900 bg-opacity-80 hover:bg-opacity-95 backdrop-blur-sm rounded-lg flex items-center justify-center transition-all border border-stone-700 border-opacity-30"
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
                                  </div>
                                </div>

                                {index < dayLocations.length - 1 && (
                                  <div className="flex justify-center -my-1 z-10 relative">
                                    <button
                                      onClick={() => {
                                        const name = prompt('Enter location name:');
                                        if (name) handleAddPlace(day, name, index);
                                      }}
                                      className="w-6 h-6 bg-zinc-800 bg-opacity-60 hover:bg-opacity-90 backdrop-blur-sm rounded-full flex items-center justify-center transition-all border border-stone-700 border-opacity-30 hover:border-orange-500 hover:border-opacity-50"
                                    >
                                      <span className="text-stone-400 hover:text-orange-400 transition-colors text-lg">+</span>
                                    </button>
                                  </div>
                                )}
                              </React.Fragment>
                            ))}

                            <div className="relative mt-3">
                              <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-stone-500 z-10">@</span>
                              <input
                                type="text"
                                value={bottomSearchInput[day] || ''}
                                placeholder="Add another place..."
                                onChange={(e) => handleBottomSearchChange(day, e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter' && bottomSearchInput[day] && bottomSearchInput[day].trim()) {
                                    handleAddPlace(day, bottomSearchInput[day].trim());
                                    setBottomSearchInput({ ...bottomSearchInput, [day]: '' });
                                    setShowBottomSuggestions({ ...showBottomSuggestions, [day]: false });
                                  }
                                }}
                                onFocus={() => {
                                  const input = bottomSearchInput[day];
                                  const suggestions = bottomSearchSuggestions[day];
                                  if (input && input.trim() && suggestions && suggestions.length > 0) {
                                    setShowBottomSuggestions({ ...showBottomSuggestions, [day]: true });
                                  }
                                }}
                                className="w-full pl-10 pr-3 py-2 bg-zinc-800 bg-opacity-40 border border-stone-700 border-opacity-30 rounded-lg text-white text-sm placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors"
                              />
                              {showBottomSuggestions[day] && bottomSearchSuggestions[day] && bottomSearchSuggestions[day].length > 0 ? (
                                <div className="absolute top-full left-0 right-0 mt-1 z-50">
                                  <SearchSuggestions
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

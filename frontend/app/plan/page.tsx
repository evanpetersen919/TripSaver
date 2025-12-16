"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Image from "next/image";

function PlanTripForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [tripName, setTripName] = useState("");
  const [destination, setDestination] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedLat, setSelectedLat] = useState("0");
  const [selectedLng, setSelectedLng] = useState("0");
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // Handle OAuth callback token and pre-fill destination
  useEffect(() => {
    const token = searchParams.get('token');
    if (token) {
      localStorage.setItem('token', token);
      // Clean URL
      window.history.replaceState({}, '', '/plan');
    }

    // Pre-fill destination from query parameter
    const location = searchParams.get('location');
    if (location) {
      console.log('Pre-filling destination:', location);
      setDestination(location);
      // Trigger search for the location
      fetch(`/api/landmarks/search?q=${encodeURIComponent(location)}`)
        .then(res => res.json())
        .then(data => {
          console.log('Search API response:', data);
          if (data.landmarks && data.landmarks.length > 0) {
            const firstResult = data.landmarks[0];
            console.log('Setting coordinates from search:', firstResult.latitude, firstResult.longitude);
            setSelectedLat(firstResult.latitude?.toString() || "0");
            setSelectedLng(firstResult.longitude?.toString() || "0");
          } else {
            console.log('No landmarks found in response');
          }
        })
        .catch(err => console.error('Error fetching location:', err));
    }
  }, [searchParams]);

  // Pre-cache popular destinations on page load
  useEffect(() => {
    const popularDestinations = [
      'Tokyo', 'Paris', 'London', 'New York', 'Rome', 
      'Barcelona', 'Bangkok', 'Dubai', 'Singapore', 'Sydney',
      'Los Angeles', 'Miami', 'San Francisco', 'Las Vegas',
      'Greece', 'Italy', 'Spain', 'Japan', 'France', 'United Kingdom'
    ];
    
    // Cache destinations in the background
    popularDestinations.forEach((dest, index) => {
      setTimeout(() => {
        fetch(`/api/landmarks/search?q=${encodeURIComponent(dest)}`).catch(() => {});
      }, index * 100); // Stagger requests to avoid rate limiting
    });
  }, []);

  const handleDestinationChange = (value: string) => {
    setDestination(value);
    setShowSuggestions(true);

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (value.trim().length < 2) {
      setSuggestions([]);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      try {
        const response = await fetch(`/api/landmarks/search?q=${encodeURIComponent(value)}`);
        const data = await response.json();
        setSuggestions(data.landmarks || []);
      } catch (error) {
        console.error('Error fetching suggestions:', error);
      }
    }, 300);
  };

  const selectSuggestion = (suggestion: any) => {
    setDestination(suggestion.name);
    setSelectedLat(suggestion.latitude?.toString() || suggestion.lat?.toString() || "0");
    setSelectedLng(suggestion.longitude?.toString() || suggestion.lng?.toString() || "0");
    setSuggestions([]);
    setShowSuggestions(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    console.log('Plan page submitting with coordinates:', { selectedLat, selectedLng, destination });
    
    const params = new URLSearchParams({
      name: tripName,
      destination: destination,
      start: startDate,
      end: endDate,
      lat: selectedLat,
      lng: selectedLng
    });
    
    router.push('/dashboard?' + params.toString());
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-900 via-stone-900 to-zinc-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Image
            src="/images/logo.png"
            alt="TripSaver"
            width={48}
            height={48}
            className="mx-auto mb-4"
            quality={100}
          />
          <h1 className="text-3xl font-bold text-white mb-2">Plan Your Trip</h1>
          <p className="text-stone-400">Create your itinerary with AI-powered recommendations</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-stone-300 mb-2">
              Trip Name
            </label>
            <input
              type="text"
              value={tripName}
              onChange={(e) => setTripName(e.target.value)}
              placeholder="e.g., Summer in Japan"
              required
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-30 rounded-lg text-white placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors"
            />
          </div>

          <div className="relative">
            <label className="block text-sm font-medium text-stone-300 mb-2">
              Destination
            </label>
            <input
              type="text"
              value={destination}
              onChange={(e) => handleDestinationChange(e.target.value)}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              placeholder="e.g., Japan, France"
              required
              spellCheck={false}
              autoComplete="off"
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-30 rounded-lg text-white placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors"
            />
            {showSuggestions && suggestions.length > 0 && (
              <div className="absolute z-50 w-full mt-1 bg-zinc-800 border border-zinc-700 rounded-lg shadow-2xl max-h-60 overflow-y-auto">
                {suggestions.slice(0, 8).map((suggestion, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => selectSuggestion(suggestion)}
                    className="w-full px-4 py-3 text-left hover:bg-zinc-700 transition-colors border-b border-zinc-700 last:border-b-0 flex items-center gap-3"
                  >
                    <svg className="w-4 h-4 text-orange-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    <div className="flex-1 min-w-0">
                      <p className="text-white text-sm font-medium truncate">{suggestion.name}</p>
                      {suggestion.country && (
                        <p className="text-stone-400 text-xs truncate">{suggestion.country}</p>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-stone-300 mb-2">
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                required
                className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-30 rounded-lg text-white focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-stone-300 mb-2">
                End Date
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                required
                className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-30 rounded-lg text-white focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors"
              />
            </div>
          </div>

          <button
            type="submit"
            className="w-full bg-orange-500 hover:bg-orange-600 text-white font-semibold py-3 px-4 rounded-lg transition-colors mt-6"
          >
            Create Trip
          </button>

          <button
            type="button"
            onClick={() => router.push('/')}
            className="w-full bg-transparent hover:bg-zinc-800 hover:bg-opacity-30 text-stone-400 hover:text-stone-300 font-medium py-3 px-4 rounded-lg transition-colors border border-stone-700 border-opacity-30"
          >
            Cancel
          </button>
        </form>
      </div>
    </div>
  );
}

export default function PlanTrip() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-gradient-to-b from-zinc-900 via-stone-900 to-zinc-900 flex items-center justify-center">Loading...</div>}>
      <PlanTripForm />
    </Suspense>
  );
}

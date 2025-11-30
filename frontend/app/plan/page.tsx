"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";

export default function PlanTrip() {
  const router = useRouter();
  const [tripName, setTripName] = useState("");
  const [destination, setDestination] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const params = new URLSearchParams({
      name: tripName,
      destination: destination,
      start: startDate,
      end: endDate,
      lat: "35.6762",
      lng: "139.6503"
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
              placeholder="e.g., Summer in Tokyo"
              required
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-30 rounded-lg text-white placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-stone-300 mb-2">
              Destination
            </label>
            <input
              type="text"
              value={destination}
              onChange={(e) => setDestination(e.target.value)}
              placeholder="e.g., Tokyo, Japan"
              required
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-30 rounded-lg text-white placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-50 transition-colors"
            />
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

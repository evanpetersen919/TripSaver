"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import Image from "next/image";

interface Trip {
  id: string;
  name: string;
  destination: string;
  startDate: string;
  endDate: string;
  locationCount: number;
  createdAt: string;
  updatedAt: string;
}

export default function Overview() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [trips, setTrips] = useState<Trip[]>([]);
  const [loading, setLoading] = useState(true);
  const [userName, setUserName] = useState("");

  useEffect(() => {
    // Handle OAuth callback token
    const urlToken = searchParams.get('token');
    const urlUsername = searchParams.get('username');
    if (urlToken) {
      localStorage.setItem('token', urlToken);
      if (urlUsername) {
        localStorage.setItem('user', JSON.stringify({ name: urlUsername }));
      }
      // Clean URL
      window.history.replaceState({}, '', '/overview');
    }

    // Check authentication
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
      return;
    }

    // Get user info
    const user = localStorage.getItem("user");
    if (user) {
      const userData = JSON.parse(user);
      setUserName(userData.name || userData.email || "User");
    }

    // Fetch user's trips
    fetchTrips();
  }, [router]);

  const fetchTrips = async () => {
    try {
      const token = localStorage.getItem("token");
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/trips`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setTrips(data.trips || []);
      }
    } catch (error) {
      console.error("Error fetching trips:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteTrip = async (tripId: string) => {
    if (!confirm("Are you sure you want to delete this trip?")) return;

    try {
      const token = localStorage.getItem("token");
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/trips/${tripId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        setTrips(trips.filter((trip) => trip.id !== tripId));
      }
    } catch (error) {
      console.error("Error deleting trip:", error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    router.push("/");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-stone-950 to-zinc-950">
      {/* Header */}
      <header className="bg-zinc-900/50 backdrop-blur-md border-b border-stone-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-8 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center">
            <Image
              src="/images/logo_text.png"
              alt="TripSaver"
              width={150}
              height={40}
              className="h-8 w-auto"
            />
          </Link>

          <div className="flex items-center gap-4">
            <span className="text-stone-400">Welcome, {userName}</span>
            <button
              onClick={handleLogout}
              className="px-4 py-2 text-stone-300 hover:text-orange-400 transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-8 py-12">
        {/* Page Title & Create Button */}
        <div className="flex items-center justify-between mb-12">
          <div>
            <h1 className="text-5xl font-bold text-stone-100 mb-2">My Trips</h1>
            <p className="text-stone-400">Plan, manage, and explore your travel itineraries</p>
          </div>
          <Link
            href="/plan"
            className="px-6 py-3 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-lg hover:shadow-lg hover:shadow-orange-500/50 transition-all duration-200 font-semibold"
          >
            + Create New Trip
          </Link>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-20">
            <div className="inline-block w-12 h-12 border-4 border-orange-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-stone-400 mt-4">Loading your trips...</p>
          </div>
        )}

        {/* Empty State */}
        {!loading && trips.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-zinc-900 flex items-center justify-center">
              <svg className="w-12 h-12 text-stone-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
              </svg>
            </div>
            <h2 className="text-2xl font-semibold text-stone-200 mb-2">No trips yet</h2>
            <p className="text-stone-500 mb-6">Start planning your next adventure!</p>
            <Link
              href="/plan"
              className="inline-block px-6 py-3 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-lg hover:shadow-lg hover:shadow-orange-500/50 transition-all duration-200 font-semibold"
            >
              Create Your First Trip
            </Link>
          </div>
        )}

        {/* Trips Grid */}
        {!loading && trips.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {trips.map((trip) => (
              <div
                key={trip.id}
                className="bg-zinc-900/50 backdrop-blur-sm rounded-xl border border-stone-800 hover:border-orange-400/50 transition-all duration-300 overflow-hidden group"
              >
                {/* Trip Header */}
                <div className="p-6">
                  <h3 className="text-xl font-semibold text-stone-100 mb-2 group-hover:text-orange-400 transition-colors">
                    {trip.name}
                  </h3>
                  <p className="text-stone-400 mb-4">{trip.destination}</p>

                  {/* Trip Details */}
                  <div className="space-y-2 mb-6">
                    <div className="flex items-center text-sm text-stone-500">
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      {new Date(trip.startDate).toLocaleDateString()} - {new Date(trip.endDate).toLocaleDateString()}
                    </div>
                    <div className="flex items-center text-sm text-stone-500">
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                      {trip.locationCount} {trip.locationCount === 1 ? 'location' : 'locations'}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-3">
                    <Link
                      href={`/dashboard?tripId=${trip.id}`}
                      className="flex-1 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors text-center font-medium"
                    >
                      Edit
                    </Link>
                    <button
                      onClick={() => handleDeleteTrip(trip.id)}
                      className="px-4 py-2 bg-zinc-800 text-stone-300 rounded-lg hover:bg-red-600 hover:text-white transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>

                {/* Trip Footer */}
                <div className="px-6 py-3 bg-zinc-950/50 border-t border-stone-800">
                  <p className="text-xs text-stone-600">
                    Last updated {new Date(trip.updatedAt).toLocaleDateString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

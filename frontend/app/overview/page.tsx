"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { DESTINATION_IMAGES, DEFAULT_DESTINATION_IMAGE } from "@/lib/destination-images";

interface Trip {
  id: string;
  name: string;
  destination: string;
  startDate: string;
  endDate: string;
  locationCount: number;
  createdAt: string;
  updatedAt: string;
  imageUrl?: string;
}

interface PopularDestination {
  name: string;
  country: string;
  photoReference?: string;
  imageUrl: string;
}

function OverviewContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [trips, setTrips] = useState<Trip[]>([]);
  const [loading, setLoading] = useState(true);
  const [userName, setUserName] = useState("");
  const [isNewUser, setIsNewUser] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [tripToDelete, setTripToDelete] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"recent" | "name" | "date">("recent");
  const [destinationImageCache, setDestinationImageCache] = useState<{ [key: string]: string }>({});
  const [popularDestinations, setPopularDestinations] = useState<PopularDestination[]>([
    { name: "Japan", country: "Asia", imageUrl: "https://images.unsplash.com/photo-1542051841857-5f90071e7989?w=800&h=600&fit=crop" },
    { name: "France", country: "Europe", imageUrl: "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=800&h=600&fit=crop" },
    { name: "Italy", country: "Europe", imageUrl: "https://images.unsplash.com/photo-1520175480921-4edfa2983e0f?w=800&h=600&fit=crop" },
    { name: "USA", country: "North America", imageUrl: "https://images.unsplash.com/photo-1518391846015-55a9cc003b25?w=800&h=600&fit=crop" },
    { name: "Spain", country: "Europe", imageUrl: "https://images.unsplash.com/photo-1562883676-8c7feb83f09b?w=800&h=600&fit=crop" },
    { name: "Greece", country: "Europe", imageUrl: "https://images.unsplash.com/photo-1555993539-1732b0258235?w=800&h=600&fit=crop" },
    { name: "Thailand", country: "Asia", imageUrl: "https://images.unsplash.com/photo-1537956965359-7573183d1f57?w=800&h=600&fit=crop" },
    { name: "Australia", country: "Oceania", imageUrl: "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?w=800&h=600&fit=crop" },
    { name: "United Kingdom", country: "Europe", imageUrl: "https://images.unsplash.com/photo-1486299267070-83823f5448dd?w=800&h=600&fit=crop" },
    { name: "Canada", country: "North America", imageUrl: "https://images.unsplash.com/photo-1517935706615-2717063c2225?w=800&h=600&fit=crop" },
    { name: "Germany", country: "Europe", imageUrl: "https://images.unsplash.com/photo-1515542622106-78bda8ba0e5b?w=800&h=600&fit=crop" },
    { name: "Iceland", country: "Europe", imageUrl: "https://images.unsplash.com/photo-1531168556467-80aace0d0144?w=800&h=600&fit=crop" },
  ]);

  // Fetch place photos from Google Places API
  const fetchPlacePhotos = async (placeName: string): Promise<string | null> => {
    try {
      const response = await fetch(`/api/landmarks/place-details?name=${encodeURIComponent(placeName)}`);
      const data = await response.json();
      if (data.photos && data.photos.length > 0) {
        return data.photos[1]?.url || data.photos[0].url;
      }
    } catch (error) {
      console.error('Error fetching place photos:', error);
    }
    return null;
  };

  // Get image for destination with Google Places API fallback
  const getDestinationImage = async (destination: string): Promise<string> => {
    // Check cache first
    if (destinationImageCache[destination]) {
      return destinationImageCache[destination];
    }

    // Check hardcoded images (same as dashboard)
    const destLower = destination.toLowerCase();
    if (DESTINATION_IMAGES[destLower]) {
      return DESTINATION_IMAGES[destLower];
    }

    // Try Google Places API as fallback
    const photoUrl = await fetchPlacePhotos(destination);
    if (photoUrl) {
      // Update cache
      setDestinationImageCache(prev => ({ ...prev, [destination]: photoUrl }));
      return photoUrl;
    }

    // Ultimate fallback
    return DEFAULT_DESTINATION_IMAGE;
  };

  // Filter and sort trips
  const filteredTrips = trips
    .filter(trip => 
      trip.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      trip.destination.toLowerCase().includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => {
      if (sortBy === "recent") return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
      if (sortBy === "name") return a.name.localeCompare(b.name);
      if (sortBy === "date") return new Date(a.startDate).getTime() - new Date(b.startDate).getTime();
      return 0;
    });

  // Calculate trip statistics
  const now = new Date();
  const upcomingTrips = trips.filter(trip => new Date(trip.startDate) > now);
  const pastTrips = trips.filter(trip => new Date(trip.endDate) < now);
  const activeTrips = trips.filter(trip => 
    new Date(trip.startDate) <= now && new Date(trip.endDate) >= now
  );

  // Get trip status
  const getTripStatus = (trip: Trip) => {
    const start = new Date(trip.startDate);
    const end = new Date(trip.endDate);
    if (start > now) return "upcoming";
    if (end < now) return "past";
    return "active";
  };

  useEffect(() => {
    // Handle OAuth callback token
    const urlToken = searchParams.get('token');
    const urlUsername = searchParams.get('username');
    const newUserParam = searchParams.get('new_user');
    
    if (urlToken) {
      localStorage.setItem('token', urlToken);
      localStorage.setItem('last_login_time', Date.now().toString());
      if (urlUsername) {
        localStorage.setItem('user', JSON.stringify({ name: urlUsername }));
      }
      // Check if this is a new user signup
      if (newUserParam === 'true') {
        setIsNewUser(true);
        localStorage.setItem('welcome_shown', 'false');
      }
      // Clean URL
      window.history.replaceState({}, '', '/overview');
    }

    // Check if user just logged in (not a refresh)
    const hasSeenWelcome = sessionStorage.getItem('welcome_seen');
    if (!hasSeenWelcome) {
      sessionStorage.setItem('welcome_seen', 'true');
      setShowWelcome(true);
    } else {
      setShowWelcome(false);
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
        const tripsData = data.trips || [];
        setTrips(tripsData);
        
        // Load images for trips that don't have them
        tripsData.forEach(async (trip: Trip) => {
          if (!trip.imageUrl && !destinationImageCache[trip.destination]) {
            const imageUrl = await getDestinationImage(trip.destination);
            setDestinationImageCache(prev => ({ ...prev, [trip.destination]: imageUrl }));
          }
        });
      }
    } catch (error) {
      console.error("Error fetching trips:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteTrip = (tripId: string) => {
    setTripToDelete(tripId);
    setShowDeleteModal(true);
  };

  const confirmDelete = async () => {
    if (!tripToDelete) return;

    try {
      const token = localStorage.getItem("token");
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/trips/${encodeURIComponent(tripToDelete)}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        fetchTrips();
      }
    } catch (error) {
      console.error("Error deleting trip:", error);
    } finally {
      setShowDeleteModal(false);
      setTripToDelete(null);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    router.push("/");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-stone-950 to-zinc-950 overflow-x-hidden">
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

      {/* Welcome Banner */}
      {showWelcome && (
        <div className="bg-zinc-950">
          <div className="max-w-7xl mx-auto px-8 py-10">
            <div className="text-center">
              {isNewUser ? (
                <>
                  <h1 className="text-4xl md:text-5xl font-bold text-stone-100 mb-3">
                    Welcome, {userName}
                  </h1>
                  <p className="text-stone-400 text-lg">
                    Your journey begins here. Create your first trip or discover new destinations.
                  </p>
                </>
              ) : (
                <>
                  <h1 className="text-4xl md:text-5xl font-bold text-stone-100 mb-3">
                    Welcome back, {userName}
                  </h1>
                  <p className="text-stone-400 text-lg">
                    Continue planning your next adventure.
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-8 py-12">
        {/* Statistics Cards */}
        {!loading && trips.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
            <div className="bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/20 rounded-xl p-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-stone-400 text-sm">Total Trips</span>
                <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
              </div>
              <p className="text-3xl font-bold text-white">{trips.length}</p>
            </div>
            <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-xl p-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-stone-400 text-sm">Upcoming</span>
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="text-3xl font-bold text-white">{upcomingTrips.length}</p>
            </div>
            <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-xl p-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-stone-400 text-sm">Active Now</span>
                <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-3xl font-bold text-white">{activeTrips.length}</p>
            </div>
            <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-xl p-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-stone-400 text-sm">Completed</span>
                <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                </svg>
              </div>
              <p className="text-3xl font-bold text-white">{pastTrips.length}</p>
            </div>
          </div>
        )}

        {/* My Locations Section */}
        <div className="mb-16">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-4xl font-bold text-stone-100 mb-2">My Locations</h2>
              <p className="text-stone-400">Your saved trips and destinations</p>
            </div>
            <Link
              href="/plan"
              className="px-8 py-3 border-2 border-orange-500 text-orange-400 rounded-xl font-semibold hover:bg-orange-500/10 hover:shadow-lg hover:shadow-orange-500/30 transition-all duration-300"
            >
              + Create New Trip
            </Link>
          </div>

          {/* Search and Filter Bar */}
          {!loading && trips.length > 0 && (
            <div className="flex flex-col md:flex-row gap-4 mb-6">
              {/* Search */}
              <div className="flex-1 relative">
                <svg className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-stone-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  placeholder="Search trips by name or destination..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-4 py-3 bg-zinc-900/50 border border-stone-800 rounded-lg text-stone-100 placeholder-stone-500 focus:outline-none focus:border-orange-500 transition-colors"
                />
              </div>

              {/* Sort Dropdown */}
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="px-4 py-3 bg-zinc-900/50 border border-stone-800 rounded-lg text-stone-100 focus:outline-none focus:border-orange-500 focus:ring-2 focus:ring-orange-500/20 transition-colors cursor-pointer appearance-none bg-no-repeat bg-right pr-10"
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23a8a29e'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
                  backgroundSize: '1.5rem',
                  backgroundPosition: 'right 0.5rem center'
                }}
              >
                <option value="recent" className="bg-zinc-900 text-stone-100">Recently Updated</option>
                <option value="name" className="bg-zinc-900 text-stone-100">Name (A-Z)</option>
                <option value="date" className="bg-zinc-900 text-stone-100">Start Date</option>
              </select>
            </div>
          )}
          {/* Loading State */}
          {loading && (
            <div className="text-center py-20">
              <div className="inline-block w-12 h-12 border-4 border-orange-500 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-stone-400 mt-4">Loading your trips...</p>
            </div>
          )}

          {/* Empty State */}
          {!loading && trips.length === 0 && (
            <div className="text-center py-12 bg-zinc-900/30 rounded-xl border border-stone-800 border-dashed">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-zinc-900 flex items-center justify-center">
                <svg className="w-8 h-8 text-stone-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-stone-300 mb-1">No saved trips</h3>
              <p className="text-stone-500 text-sm">Create your first trip to get started</p>
            </div>
          )}

          {/* No Results */}
          {!loading && trips.length > 0 && filteredTrips.length === 0 && (
            <div className="text-center py-12 bg-zinc-900/30 rounded-xl border border-stone-800">
              <svg className="w-16 h-16 mx-auto mb-4 text-stone-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <h3 className="text-xl font-semibold text-stone-300 mb-1">No trips found</h3>
              <p className="text-stone-500 text-sm">Try adjusting your search or filters</p>
            </div>
          )}

          {/* Trips Grid */}
          {!loading && filteredTrips.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredTrips.map((trip) => (
              <div
                key={trip.id}
                className="bg-zinc-900/50 backdrop-blur-sm rounded-xl border border-stone-800 hover:border-orange-400/50 transition-all duration-300 overflow-hidden group"
              >
                {/* Trip Image */}
                <div className="relative h-48 overflow-hidden bg-black">
                  <Image
                    src={DESTINATION_IMAGES[trip.destination.toLowerCase()] || destinationImageCache[trip.destination] || trip.imageUrl || DEFAULT_DESTINATION_IMAGE}
                    alt={trip.destination}
                    fill
                    className="object-cover group-hover:scale-110 transition-transform duration-500"
                    placeholder="empty"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-zinc-900 via-zinc-900/40 to-transparent"></div>
                </div>
                
                {/* Trip Header */}
                <div className="p-6">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="text-xl font-semibold text-stone-100 group-hover:text-orange-400 transition-colors flex-1">
                      {trip.name}
                    </h3>
                    {getTripStatus(trip) === "upcoming" && (
                      <span className="ml-2 px-2 py-1 text-xs font-semibold bg-blue-500/20 text-blue-400 rounded-full">Upcoming</span>
                    )}
                    {getTripStatus(trip) === "active" && (
                      <span className="ml-2 px-2 py-1 text-xs font-semibold bg-green-500/20 text-green-400 rounded-full">Active</span>
                    )}
                    {getTripStatus(trip) === "past" && (
                      <span className="ml-2 px-2 py-1 text-xs font-semibold bg-stone-500/20 text-stone-400 rounded-full">Completed</span>
                    )}
                  </div>
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
                      href={`/dashboard?tripId=${encodeURIComponent(trip.id)}`}
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

        {/* Discover New Locations Section */}
        <div className="mb-12">
          <div className="mb-8">
            <h2 className="text-4xl font-bold text-stone-100 mb-2">Discover New Locations</h2>
            <p className="text-stone-400">Explore popular destinations around the world</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {popularDestinations.map((destination) => (
              <Link
                key={destination.name}
                href={`/plan?location=${encodeURIComponent(destination.name)}`}
                className="group relative overflow-hidden rounded-xl border border-stone-800 hover:border-orange-400 transition-all duration-300 cursor-pointer"
              >
                {/* Destination Image */}
                <div className="relative h-64 overflow-hidden">
                  <img
                    src={destination.imageUrl}
                    alt={destination.name}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
                  
                  {/* Destination Info */}
                  <div className="absolute bottom-0 left-0 right-0 p-6">
                    <h3 className="text-2xl font-bold text-white mb-1 group-hover:text-orange-400 transition-colors">
                      {destination.name}
                    </h3>
                    <p className="text-stone-300 text-sm flex items-center">
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                      {destination.country}
                    </p>
                  </div>

                  {/* Hover Overlay */}
                  <div className="absolute inset-0 bg-orange-500/0 group-hover:bg-orange-500/10 transition-all duration-300 flex items-center justify-center">
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-white/90 rounded-full p-3">
                      <svg className="w-6 h-6 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                      </svg>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 border border-stone-800 rounded-2xl p-8 max-w-md w-full shadow-2xl">
            <div className="mb-6">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center">
                <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white text-center mb-2">Delete Trip?</h3>
              <p className="text-stone-400 text-center">This action cannot be undone. All trip data will be permanently deleted.</p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowDeleteModal(false);
                  setTripToDelete(null);
                }}
                className="flex-1 px-6 py-3 bg-zinc-800 text-stone-300 rounded-lg hover:bg-zinc-700 transition-colors font-semibold"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="flex-1 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-semibold"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 border border-stone-800 rounded-2xl p-8 max-w-md w-full shadow-2xl">
            <div className="mb-6">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center">
                <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white text-center mb-2">Delete Trip?</h3>
              <p className="text-stone-400 text-center">This action cannot be undone. All trip data will be permanently deleted.</p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowDeleteModal(false);
                  setTripToDelete(null);
                }}
                className="flex-1 px-6 py-3 bg-zinc-800 text-stone-300 rounded-lg hover:bg-zinc-700 transition-colors font-semibold"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="flex-1 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-semibold"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Overview() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-zinc-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-orange-500 border-r-transparent mb-4"></div>
          <p className="text-zinc-400">Loading your trips...</p>
        </div>
      </div>
    }>
      <OverviewContent />
    </Suspense>
  );
}
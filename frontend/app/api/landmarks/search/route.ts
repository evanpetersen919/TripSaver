import { NextResponse } from 'next/server';

// Simple in-memory cache with 5 minute TTL
const cache = new Map<string, { data: any[], timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('q');
  const country = searchParams.get('country'); // Optional country filter

  if (!query || query.trim().length < 2) {
    return NextResponse.json({ landmarks: [] });
  }

  // Check cache first (include country in cache key)
  const cacheKey = country ? `${query.toLowerCase()}|${country.toLowerCase()}` : query.toLowerCase();
  const cached = cache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return NextResponse.json({ landmarks: cached.data });
  }

  try {
    // Use Nominatim (OpenStreetMap) API for location search with English language preference
    const searchQuery = country ? `${query}, ${country}` : query;
    const nominatimUrl = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(searchQuery)}&format=json&limit=10&addressdetails=1&accept-language=en`;
    
    const response = await fetch(nominatimUrl, {
      headers: {
        'User-Agent': 'TripSaver-Dashboard/1.0', // Required by Nominatim
        'Accept-Language': 'en'
      },
      next: { revalidate: 300 } // Cache for 5 minutes in Next.js
    });

    if (!response.ok) {
      throw new Error('Nominatim API request failed');
    }

    const data = await response.json();
    
    let results = data.map((location: any) => {
      // Extract clean English name
      let name = location.display_name.split(',')[0].trim();
      
      // Fallback to other name fields if display_name is not useful
      if (location.name && location.name !== name) {
        name = location.name;
      }
      
      return {
        name: name,
        latitude: parseFloat(location.lat),
        longitude: parseFloat(location.lon),
        country: location.address?.country || '',
        description: location.display_name,
        type: location.type || '',
        addresstype: location.addresstype || ''
      };
    });

    // Filter by country if specified (additional client-side filter)
    if (country) {
      results = results.filter((r: any) => 
        r.country.toLowerCase().includes(country.toLowerCase())
      );
    }

    // Remove duplicates by name (case-insensitive)
    const seen = new Set<string>();
    results = results.filter((r: any) => {
      const nameLower = r.name.toLowerCase();
      if (seen.has(nameLower)) {
        return false;
      }
      seen.add(nameLower);
      return true;
    });

    // Store in cache
    cache.set(cacheKey, {
      data: results,
      timestamp: Date.now()
    });

    return NextResponse.json({ landmarks: results });
  } catch (error) {
    console.error('Error fetching from Nominatim:', error);
    return NextResponse.json({ landmarks: [], error: 'Failed to search locations' }, { status: 500 });
  }
}

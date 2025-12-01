import { NextRequest, NextResponse } from 'next/server';

const GOOGLE_MAPS_API_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const query = searchParams.get('q');

  if (!query || query.trim().length < 2) {
    return NextResponse.json({ landmarks: [] });
  }

  try {
    // Use new Places API Text Search
    const searchUrl = `https://places.googleapis.com/v1/places:searchText`;
    
    const response = await fetch(searchUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
        'X-Goog-FieldMask': 'places.displayName,places.id,places.location,places.formattedAddress,places.types,places.photos'
      },
      body: JSON.stringify({
        textQuery: query,
        maxResultCount: 10
      })
    });

    const data = await response.json();
    
    // Check for API errors
    if (data.error) {
      console.error('Google Places API error:', data.error);
      // Fallback to old search API
      const fallbackResponse = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=10&addressdetails=1&accept-language=en`, {
        headers: {
          'User-Agent': 'TripSaver-Dashboard/1.0',
          'Accept-Language': 'en'
        }
      });
      const fallbackData = await fallbackResponse.json();
      const landmarks = fallbackData.map((location: any) => ({
        name: location.display_name.split(',')[0].trim(),
        latitude: parseFloat(location.lat),
        longitude: parseFloat(location.lon),
        description: location.display_name,
        types: [],
        photo: null
      }));
      return NextResponse.json({ landmarks });
    }

    if (!data.places || data.places.length === 0) {
      return NextResponse.json({ landmarks: [] });
    }

    // Transform to match expected format
    const landmarks = data.places.map((place: any) => ({
      name: place.displayName?.text || '',
      latitude: place.location?.latitude || 0,
      longitude: place.location?.longitude || 0,
      description: place.formattedAddress || '',
      types: place.types || [],
      photo: place.photos?.[0] ? `https://places.googleapis.com/v1/${place.photos[0].name}/media?key=${GOOGLE_MAPS_API_KEY}&maxWidthPx=400` : null
    }));

    return NextResponse.json({ landmarks });

  } catch (error) {
    console.error('Error fetching from Google Places:', error);
    return NextResponse.json({ landmarks: [], error: 'Failed to search locations' }, { status: 500 });
  }
}

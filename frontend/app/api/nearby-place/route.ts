import { NextRequest, NextResponse } from 'next/server';

const GOOGLE_MAPS_API_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const lat = searchParams.get('lat');
  const lng = searchParams.get('lng');

  if (!lat || !lng) {
    return NextResponse.json({ error: 'Latitude and longitude are required' }, { status: 400 });
  }

  try {
    // Use new Places API (Text Search) to find places at the clicked location
    const searchUrl = `https://places.googleapis.com/v1/places:searchNearby`;
    
    const response = await fetch(searchUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
        'X-Goog-FieldMask': 'places.displayName,places.id,places.location,places.types'
      },
      body: JSON.stringify({
        locationRestriction: {
          circle: {
            center: {
              latitude: parseFloat(lat),
              longitude: parseFloat(lng)
            },
            radius: 50.0
          }
        },
        maxResultCount: 1
      })
    });
    
    const data = await response.json();

    if (!data.places || data.places.length === 0) {
      return NextResponse.json({ error: 'No place found at this location' }, { status: 404 });
    }

    // Return the first place
    const place = data.places[0];
    
    return NextResponse.json({
      name: place.displayName?.text || 'Unknown Place',
      lat: place.location?.latitude || parseFloat(lat),
      lng: place.location?.longitude || parseFloat(lng),
      placeId: place.id,
      types: place.types || []
    });

  } catch (error) {
    console.error('Error fetching nearby place:', error);
    return NextResponse.json(
      { error: 'Failed to fetch nearby place', details: String(error) },
      { status: 500 }
    );
  }
}

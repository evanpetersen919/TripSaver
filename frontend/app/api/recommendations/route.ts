import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { destinationLat, destinationLng, existingLocations } = body;

    console.log('Recommendations API received:', { destinationLat, destinationLng, type: typeof destinationLat });

    if (!destinationLat || !destinationLng || isNaN(destinationLat) || isNaN(destinationLng)) {
      console.error('Invalid coordinates:', { destinationLat, destinationLng });
      return NextResponse.json(
        { error: 'Destination coordinates required', received: { destinationLat, destinationLng } },
        { status: 400 }
      );
    }

    // Use Google Places API to find nearby tourist attractions
    const response = await fetch(
      `https://places.googleapis.com/v1/places:searchNearby`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Goog-Api-Key': process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '',
          'X-Goog-FieldMask': 'places.displayName,places.location'
        },
        body: JSON.stringify({
          includedTypes: ['tourist_attraction', 'museum', 'church', 'park', 'art_gallery', 'historical_landmark'],
          maxResultCount: 10,
          locationRestriction: {
            circle: {
              center: {
                latitude: parseFloat(destinationLat),
                longitude: parseFloat(destinationLng)
              },
              radius: 10000.0
            }
          }
        })
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Google Places API error:', errorText);
      return NextResponse.json(
        { error: 'Failed to fetch nearby places' },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    // Filter out locations already in itinerary
    const existingNames = new Set(
      (existingLocations || []).map((loc: any) => loc.name.toLowerCase())
    );
    
    const recommendations = (data.places || [])
      .filter((place: any) => !existingNames.has(place.displayName?.text?.toLowerCase() || ''))
      .map((place: any, index: number) => ({
        name: place.displayName?.text || 'Unknown Location',
        lat: place.location?.latitude || destinationLat,
        lng: place.location?.longitude || destinationLng,
        confidence: 0.9 - (index * 0.05)
      }))
      .slice(0, 5);

    return NextResponse.json({ recommendations });

  } catch (error) {
    console.error('Error generating recommendations:', error);
    return NextResponse.json(
      { error: 'Failed to generate recommendations' },
      { status: 500 }
    );
  }
}

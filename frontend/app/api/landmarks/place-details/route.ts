import { NextRequest, NextResponse } from 'next/server';

const GOOGLE_MAPS_API_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const landmarkName = searchParams.get('name');

  if (!landmarkName) {
    return NextResponse.json({ error: 'Landmark name required' }, { status: 400 });
  }

  console.log(`[PlaceDetails] Fetching details for: ${landmarkName}`);

  try {
    // First, search for the place to get place_id
    const searchUrl = `https://places.googleapis.com/v1/places:searchText`;
    
    const searchResponse = await fetch(searchUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
        'X-Goog-FieldMask': 'places.id'
      },
      body: JSON.stringify({
        textQuery: landmarkName,
        maxResultCount: 1
      })
    });

    const searchData = await searchResponse.json();
    console.log(`[PlaceDetails] Search response:`, searchData);

    if (searchData.error) {
      console.error('[PlaceDetails] Search API error:', searchData.error);
      return NextResponse.json({ photos: [], location: null, error: searchData.error });
    }

    if (!searchData.places || searchData.places.length === 0) {
      console.log('[PlaceDetails] No places found');
      return NextResponse.json({ photos: [], location: null });
    }

    const placeId = searchData.places[0].id;
    console.log(`[PlaceDetails] Found place ID: ${placeId}`);

    // Now get place details with photos
    const detailsUrl = `https://places.googleapis.com/v1/places/${placeId}`;
    
    const detailsResponse = await fetch(detailsUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
        'X-Goog-FieldMask': 'photos,location,addressComponents'
      }
    });

    const detailsData = await detailsResponse.json();
    console.log(`[PlaceDetails] Details response:`, detailsData);

    if (detailsData.error) {
      console.error('[PlaceDetails] Details API error:', detailsData.error);
      return NextResponse.json({ photos: [], location: null, city: null, country: null, error: detailsData.error });
    }

    // Extract city and country from address components
    let city = null;
    let country = null;
    
    if (detailsData.addressComponents) {
      for (const component of detailsData.addressComponents) {
        if (component.types.includes('locality')) {
          city = component.longText;
        }
        if (component.types.includes('country')) {
          country = component.longText;
        }
      }
    }

    if (!detailsData.photos || detailsData.photos.length === 0) {
      console.log('[PlaceDetails] No photos found');
      return NextResponse.json({ photos: [], location: detailsData.location || null, city, country });
    }

    // Return up to 3 photos with full URLs
    const photos = detailsData.photos.slice(0, 3).map((photo: any) => ({
      name: photo.name,
      url: `https://places.googleapis.com/v1/${photo.name}/media?key=${GOOGLE_MAPS_API_KEY}&maxWidthPx=800`
    }));

    console.log(`[PlaceDetails] Returning ${photos.length} photos, city: ${city}, country: ${country}`);

    return NextResponse.json({ 
      photos,
      location: detailsData.location || null,
      city,
      country
    });

  } catch (error) {
    console.error('[PlaceDetails] Error fetching place details:', error);
    return NextResponse.json({ photos: [], location: null, error: String(error) });
  }
}

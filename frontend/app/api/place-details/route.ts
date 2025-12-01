import { NextRequest, NextResponse } from 'next/server';

const GOOGLE_MAPS_API_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const placeName = searchParams.get('name');
  const lat = searchParams.get('lat');
  const lng = searchParams.get('lng');

  if (!placeName || !lat || !lng) {
    return NextResponse.json({ error: 'Place name, lat, and lng are required' }, { status: 400 });
  }

  try {
    // Step 1: Find the place using new Places API Text Search
    const searchUrl = `https://places.googleapis.com/v1/places:searchText`;
    
    const searchResponse = await fetch(searchUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
        'X-Goog-FieldMask': 'places.id,places.displayName'
      },
      body: JSON.stringify({
        textQuery: placeName,
        locationBias: {
          circle: {
            center: {
              latitude: parseFloat(lat),
              longitude: parseFloat(lng)
            },
            radius: 500.0
          }
        }
      })
    });

    const searchData = await searchResponse.json();

    if (!searchData.places || searchData.places.length === 0) {
      // Fallback to basic data if place not found
      return NextResponse.json({
        name: placeName,
        rating: null,
        totalRatings: 0,
        description: '',
        openingHours: '',
        website: '',
        phone: '',
        photos: [],
        reviews: [],
        priceLevel: null,
        types: [],
        address: ''
      });
    }

    const placeId = searchData.places[0].id;

    // Step 2: Get detailed information using new Place Details API
    const detailsUrl = `https://places.googleapis.com/v1/${placeId}`;
    
    const detailsResponse = await fetch(detailsUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
        'X-Goog-FieldMask': 'displayName,rating,userRatingCount,formattedAddress,internationalPhoneNumber,websiteUri,regularOpeningHours,priceLevel,reviews,photos,types,editorialSummary,currentOpeningHours'
      }
    });

    const detailsData = await detailsResponse.json();

    if (!detailsData.displayName) {
      return NextResponse.json({ error: 'Failed to fetch place details' }, { status: 500 });
    }

    // Extract photos (up to 5)
    const photos = (detailsData.photos || []).slice(0, 5).map((photo: any) => {
      const photoName = photo.name;
      return `https://places.googleapis.com/v1/${photoName}/media?key=${GOOGLE_MAPS_API_KEY}&maxWidthPx=800`;
    });

    // Extract reviews (up to 5)
    const reviews = (detailsData.reviews || []).slice(0, 5).map((review: any) => ({
      author: review.authorAttribution?.displayName || 'Anonymous',
      rating: review.rating || 0,
      text: review.text?.text || '',
      date: review.relativePublishTimeDescription || '',
      profilePhoto: review.authorAttribution?.photoUri || ''
    }));

    // Format opening hours
    let openingHours = '';
    if (detailsData.regularOpeningHours?.weekdayDescriptions) {
      openingHours = detailsData.regularOpeningHours.weekdayDescriptions.join('\n');
    }

    return NextResponse.json({
      name: detailsData.displayName?.text || placeName,
      rating: detailsData.rating || null,
      totalRatings: detailsData.userRatingCount || 0,
      description: detailsData.editorialSummary?.text || '',
      address: detailsData.formattedAddress || '',
      openingHours: openingHours,
      website: detailsData.websiteUri || '',
      phone: detailsData.internationalPhoneNumber || '',
      photos: photos,
      reviews: reviews,
      priceLevel: detailsData.priceLevel ? ['FREE', 'INEXPENSIVE', 'MODERATE', 'EXPENSIVE', 'VERY_EXPENSIVE'].indexOf(detailsData.priceLevel) : null,
      types: detailsData.types || [],
      isOpen: detailsData.currentOpeningHours?.openNow
    });

  } catch (error) {
    console.error('Error fetching place details:', error);
    return NextResponse.json(
      { error: 'Failed to fetch place details' },
      { status: 500 }
    );
  }
}

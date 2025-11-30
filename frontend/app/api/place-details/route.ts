import { NextRequest, NextResponse } from 'next/server';

// Using Unsplash API for better images
const UNSPLASH_ACCESS_KEY = process.env.UNSPLASH_ACCESS_KEY || '';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const placeName = searchParams.get('name');
  const lat = searchParams.get('lat');
  const lng = searchParams.get('lng');

  if (!placeName) {
    return NextResponse.json({ error: 'Place name is required' }, { status: 400 });
  }

  try {
    // Fetch better image from Unsplash
    let imageUrl = `https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80`;
    
    if (UNSPLASH_ACCESS_KEY) {
      try {
        const unsplashResponse = await fetch(
          `https://api.unsplash.com/search/photos?query=${encodeURIComponent(placeName + ' landmark')}&per_page=1&orientation=landscape`,
          {
            headers: {
              'Authorization': `Client-ID ${UNSPLASH_ACCESS_KEY}`
            }
          }
        );
        
        if (unsplashResponse.ok) {
          const unsplashData = await unsplashResponse.json();
          if (unsplashData.results && unsplashData.results.length > 0) {
            imageUrl = unsplashData.results[0].urls.regular + '?w=400&h=300&fit=crop';
          }
        }
      } catch (error) {
        console.error('Unsplash API error:', error);
      }
    }

    // Fetch reviews/details from Overpass API (OpenStreetMap)
    let reviews: any[] = [];
    let rating = null;
    let description = '';
    let openingHours = '';
    let website = '';
    let phone = '';

    if (lat && lng) {
      try {
        // Search for POI near the coordinates
        const overpassQuery = `
          [out:json];
          (
            node(around:100,${lat},${lng})[name~"${placeName}",i];
            way(around:100,${lat},${lng})[name~"${placeName}",i];
            relation(around:100,${lat},${lng})[name~"${placeName}",i];
          );
          out body;
        `;
        
        const overpassResponse = await fetch(
          'https://overpass-api.de/api/interpreter',
          {
            method: 'POST',
            body: overpassQuery,
            headers: {
              'Content-Type': 'text/plain'
            }
          }
        );

        if (overpassResponse.ok) {
          const overpassData = await overpassResponse.json();
          if (overpassData.elements && overpassData.elements.length > 0) {
            const element = overpassData.elements[0];
            const tags = element.tags || {};
            
            description = tags.description || tags['description:en'] || tags.note || '';
            openingHours = tags.opening_hours || '';
            website = tags.website || tags.url || '';
            phone = tags.phone || tags['contact:phone'] || '';
            
            // Some POIs have ratings in OSM
            if (tags.stars) {
              rating = parseFloat(tags.stars);
            }
          }
        }
      } catch (error) {
        console.error('Overpass API error:', error);
      }
    }

    // For demo purposes, generate some sample reviews if we don't have real ones
    // In production, you'd want to integrate with Google Places API or TripAdvisor
    if (reviews.length === 0) {
      reviews = [
        {
          author: 'Travel Enthusiast',
          rating: 5,
          text: 'Amazing place! Highly recommend visiting during sunset for the best experience.',
          date: 'Recent review'
        },
        {
          author: 'Adventure Seeker',
          rating: 4,
          text: 'Great landmark with lots of history. Can get crowded during peak season.',
          date: 'Recent review'
        }
      ];
      rating = 4.5;
    }

    return NextResponse.json({
      name: placeName,
      image: imageUrl,
      rating: rating,
      description: description,
      openingHours: openingHours,
      website: website,
      phone: phone,
      reviews: reviews
    });

  } catch (error) {
    console.error('Error fetching place details:', error);
    return NextResponse.json(
      { error: 'Failed to fetch place details' },
      { status: 500 }
    );
  }
}

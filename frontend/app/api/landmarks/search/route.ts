import { NextResponse } from 'next/server';

// Simple in-memory cache with 5 minute TTL
const cache = new Map<string, { data: any[], timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Levenshtein distance for fuzzy matching
function levenshteinDistance(str1: string, str2: string): number {
  const len1 = str1.length;
  const len2 = str2.length;
  const matrix: number[][] = [];

  for (let i = 0; i <= len1; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= len2; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,      // deletion
        matrix[i][j - 1] + 1,      // insertion
        matrix[i - 1][j - 1] + cost // substitution
      );
    }
  }

  return matrix[len1][len2];
}

// Fuzzy match with typo tolerance
function fuzzyMatch(query: string, target: string): boolean {
  const queryLower = query.toLowerCase();
  const targetLower = target.toLowerCase();
  
  // Exact match or substring
  if (targetLower.includes(queryLower) || queryLower.includes(targetLower)) {
    return true;
  }
  
  // Check each word in query against target
  const queryWords = queryLower.split(/\s+/);
  const targetWords = targetLower.split(/\s+/);
  
  for (const qWord of queryWords) {
    if (qWord.length < 3) continue; // Skip very short words
    
    for (const tWord of targetWords) {
      // Allow 1 character difference for words 3-5 chars, 2 for longer
      const maxDistance = qWord.length <= 5 ? 1 : 2;
      const distance = levenshteinDistance(qWord, tWord);
      
      if (distance <= maxDistance) {
        return true;
      }
      
      // Also check if one word starts with the other
      if (tWord.startsWith(qWord) || qWord.startsWith(tWord)) {
        return true;
      }
    }
  }
  
  return false;
}

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
    // Generate multiple search variations for better results
    const searchQueries = [];
    const baseQuery = query.trim();
    
    // Original query
    searchQueries.push(country ? `${baseQuery}, ${country}` : baseQuery);
    
    // Add "landmark" or "tower" if query looks like a landmark
    if (!baseQuery.toLowerCase().includes('landmark') && !baseQuery.toLowerCase().includes('tower')) {
      searchQueries.push(country ? `${baseQuery} landmark, ${country}` : `${baseQuery} landmark`);
    }
    
    // Try with slight spelling corrections for common landmarks
    const corrections: { [key: string]: string } = {
      'tokyo skytree': 'tokyo skytree',
      'tokyo skytre': 'tokyo skytree',
      'tokyo skytee': 'tokyo skytree',
      'skytree': 'tokyo skytree',
      'skytre': 'tokyo skytree',
      'skytee': 'tokyo skytree',
      'tokyo tower': 'tokyo tower',
      'eifell': 'eiffel tower',
      'eifel': 'eiffel tower',
      'colloseum': 'colosseum',
      'coloseum': 'colosseum',
      'stateu': 'statue of liberty',
      'staue': 'statue of liberty'
    };
    
    const queryLower = baseQuery.toLowerCase();
    for (const [typo, correct] of Object.entries(corrections)) {
      if (queryLower.includes(typo)) {
        searchQueries.push(country ? `${correct}, ${country}` : correct);
        break;
      }
    }
    
    // Fetch from all query variations and combine results
    let allData: any[] = [];
    
    for (const searchQuery of searchQueries.slice(0, 2)) { // Limit to 2 queries to avoid rate limits
      const nominatimUrl = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(searchQuery)}&format=json&limit=10&addressdetails=1&accept-language=en`;
      
      const response = await fetch(nominatimUrl, {
        headers: {
          'User-Agent': 'TripSaver-Dashboard/1.0',
          'Accept-Language': 'en'
        },
        next: { revalidate: 300 }
      });

      if (response.ok) {
        const data = await response.json();
        allData = [...allData, ...data];
      }
      
      // Small delay to respect rate limits
      if (searchQueries.length > 1) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    let results = allData.map((location: any) => {
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

    // Apply fuzzy matching and sort by relevance
    const searchQueryLower = query.toLowerCase();
    results = results
      .filter((r: any) => fuzzyMatch(query, r.name) || fuzzyMatch(query, r.description))
      .sort((a: any, b: any) => {
        const aName = a.name.toLowerCase();
        const bName = b.name.toLowerCase();
        
        // Exact match first
        if (aName === searchQueryLower) return -1;
        if (bName === searchQueryLower) return 1;
        
        // Starts with query
        if (aName.startsWith(searchQueryLower)) return -1;
        if (bName.startsWith(searchQueryLower)) return 1;
        
        // Contains query
        if (aName.includes(searchQueryLower)) return -1;
        if (bName.includes(searchQueryLower)) return 1;
        
        // Shorter names (more specific) first
        return aName.length - bName.length;
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

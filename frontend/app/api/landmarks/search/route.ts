import { NextResponse } from 'next/server';

// Simple in-memory cache with 24 hour TTL (landmark data doesn't change often)
const cache = new Map<string, { data: any[], timestamp: number }>();
const CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours

// Request deduplication to prevent multiple simultaneous requests for same query
const pendingRequests = new Map<string, Promise<any>>();

// Optimized Levenshtein distance - only for short strings (< 15 chars)
function levenshteinDistance(str1: string, str2: string): number {
  const len1 = str1.length;
  const len2 = str2.length;
  
  // Skip expensive calculation for long strings or very different lengths
  if (len1 > 15 || len2 > 15 || Math.abs(len1 - len2) > 3) {
    return 999;
  }
  
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

// Optimized fuzzy match with early exits
function fuzzyMatch(query: string, target: string): boolean {
  const queryLower = query.toLowerCase();
  const targetLower = target.toLowerCase();
  
  // Fast path: exact match or substring (90% of cases)
  if (targetLower.includes(queryLower) || queryLower.includes(targetLower)) {
    return true;
  }
  
  // Check each word in query against target
  const queryWords = queryLower.split(/\s+/);
  const targetWords = targetLower.split(/\s+/);
  
  for (const qWord of queryWords) {
    if (qWord.length < 3) continue; // Skip very short words
    
    for (const tWord of targetWords) {
      // Fast path: check prefix matching first (cheaper than Levenshtein)
      if (tWord.startsWith(qWord) || qWord.startsWith(tWord)) {
        return true;
      }
      
      // Only do expensive Levenshtein for reasonable word lengths
      if (qWord.length <= 15 && tWord.length <= 15) {
        const maxDistance = qWord.length <= 5 ? 1 : 2;
        const distance = levenshteinDistance(qWord, tWord);
        
        if (distance <= maxDistance) {
          return true;
        }
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
  
  // Request deduplication - if same request is in flight, wait for it
  const pendingRequest = pendingRequests.get(cacheKey);
  if (pendingRequest) {
    const data = await pendingRequest;
    return NextResponse.json({ landmarks: data });
  }

  // Create promise for request deduplication
  const requestPromise = (async () => {
    try {
      const baseQuery = query.trim();
      const baseQueryLower = baseQuery.toLowerCase();
    

    
    // OPTIMIZATION: Only make ONE API call instead of 2 (cuts latency in half)
    // Build optimal query with country context
    const searchQuery = country ? `${baseQuery}, ${country}` : baseQuery;
    
    // Try spelling corrections for common typos
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
    
    let finalQuery = searchQuery;
    for (const [typo, correct] of Object.entries(corrections)) {
      if (baseQueryLower.includes(typo)) {
        finalQuery = country ? `${correct}, ${country}` : correct;
        break;
      }
    }
    
    const nominatimUrl = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(finalQuery)}&format=json&limit=20&addressdetails=1&accept-language=en`;
    
    const response = await fetch(nominatimUrl, {
      headers: {
        'User-Agent': 'TripSaver-Dashboard/1.0',
        'Accept-Language': 'en'
      },
      next: { revalidate: 86400 } // Cache for 24 hours
    });

    if (!response.ok) {
      throw new Error('Nominatim API request failed');
    }
    
    const allData = await response.json();
    
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

    // Smart filtering and scoring
    const searchQueryLower = query.toLowerCase();
    
    // Filter with fuzzy matching
    results = results.filter((r: any) => 
      fuzzyMatch(query, r.name) || fuzzyMatch(query, r.description)
    );
    
    // Score and sort results intelligently
    results = results.map((r: any) => {
      const nameLower = r.name.toLowerCase();
      let score = 0;
      
      // Exact match: highest priority
      if (nameLower === searchQueryLower) score += 1000;
      
      // Starts with query: high priority
      else if (nameLower.startsWith(searchQueryLower)) score += 500;
      
      // Contains query: medium priority  
      else if (nameLower.includes(searchQueryLower)) score += 250;
      
      // Word-level matching bonus (e.g., "tokyo tower" should beat "tokyo")
      const queryWords = searchQueryLower.split(' ');
      const nameWords = nameLower.split(' ');
      if (queryWords.length > 1) {
        // Multi-word query: bonus if all words present
        const allWordsMatch = queryWords.every((qw: string) => nameWords.some((nw: string) => nw.startsWith(qw) || nw.includes(qw)));
        if (allWordsMatch) score += 300;
      }
      
      // Bonus for landmark/tourism types
      if (r.type === 'tourism' || r.addresstype === 'tourism') score += 100;
      if (/monument|memorial|attraction|landmark|tower|museum|castle|palace|temple|cathedral/i.test(r.type)) score += 150;
      
      // Penalty for generic location types
      if (/street|road|neighbourhood|suburb|city_block|administrative|region|state|country/i.test(r.addresstype)) score -= 100;
      if (r.addresstype === 'city' && queryWords.length > 1) score -= 50; // Penalize cities when searching for specific landmarks
      
      // Penalty for very long names (likely addresses)
      if (nameLower.length > 50) score -= 100;
      
      // Bonus for matching country context
      if (country && r.country.toLowerCase().includes(country.toLowerCase())) score += 75;
      
      return { ...r, score };
    }).sort((a: any, b: any) => b.score - a.score);

      // Store in cache
      cache.set(cacheKey, {
        data: results,
        timestamp: Date.now()
      });

      return results;
    } catch (error) {
      console.error('Error fetching from Nominatim:', error);
      throw error;
    }
  })();
  
  // Store pending request for deduplication
  pendingRequests.set(cacheKey, requestPromise);
  
  try {
    const results = await requestPromise;
    return NextResponse.json({ landmarks: results });
  } catch (error) {
    return NextResponse.json({ landmarks: [], error: 'Failed to search locations' }, { status: 500 });
  } finally {
    // Clean up pending request
    pendingRequests.delete(cacheKey);
  }
}

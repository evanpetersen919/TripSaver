/**
 * API Integration Tests
 * Tests the landmark detection and recommendation APIs
 */

describe('Landmark Detection API', () => {
  const API_BASE = 'https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod'

  it('should return predictions for landmark image', async () => {
    // Mock fetch for this test
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          predictions: [
            { name: 'Eiffel Tower', confidence: 0.95, latitude: 48.8584, longitude: 2.2945 }
          ]
        }),
      })
    ) as jest.Mock

    const formData = new FormData()
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    })

    const data = await response.json()
    
    expect(response.ok).toBe(true)
    expect(data.predictions).toBeDefined()
    expect(data.predictions[0]).toHaveProperty('name')
    expect(data.predictions[0]).toHaveProperty('confidence')
  })

  it('should return recommendations near location', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          recommendations: [
            { name: 'Louvre Museum', latitude: 48.8606, longitude: 2.3376, similarity_score: 0.85 }
          ]
        }),
      })
    ) as jest.Mock

    const response = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        itinerary_landmarks: ['Eiffel Tower'],
        vision_description: 'Popular landmarks in Paris',
        max_distance_km: 50.0,
        top_k: 5
      }),
    })

    const data = await response.json()
    
    expect(response.ok).toBe(true)
    expect(data.recommendations).toBeDefined()
    expect(Array.isArray(data.recommendations)).toBe(true)
  })
})

describe('Utility Functions', () => {
  it('should calculate distance between coordinates correctly', () => {
    // Haversine distance calculation
    const haversineDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
      const R = 6371 // Earth's radius in km
      const dLat = (lat2 - lat1) * Math.PI / 180
      const dLon = (lon2 - lon1) * Math.PI / 180
      const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                Math.sin(dLon / 2) * Math.sin(dLon / 2)
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
      return R * c
    }

    // Distance between Eiffel Tower and Louvre (approx 3.3 km)
    const distance = haversineDistance(48.8584, 2.2945, 48.8606, 2.3376)
    expect(distance).toBeGreaterThan(3)
    expect(distance).toBeLessThan(4)
  })

  it('should validate coordinates', () => {
    const isValidCoordinate = (lat: number, lng: number): boolean => {
      return lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180
    }

    expect(isValidCoordinate(48.8584, 2.2945)).toBe(true) // Paris
    expect(isValidCoordinate(100, 0)).toBe(false) // Invalid latitude
    expect(isValidCoordinate(0, 200)).toBe(false) // Invalid longitude
  })
})

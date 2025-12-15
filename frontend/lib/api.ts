const API_BASE_URL = 'https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod';

export interface LandmarkPrediction {
  landmark: string;
  confidence: number;
  location?: string;
  latitude?: number;
  longitude?: number;
}

export interface PredictResponse {
  success: boolean;
  predictions: LandmarkPrediction[];
  llava_description?: string;
  confidence_level: string;
  strategy: string;
  error?: string;
}

export interface AuthResponse {
  success: boolean;
  access_token?: string;
  message?: string;
  user_id?: string;
}

export interface ItineraryItem {
  itinerary_id: string;
  landmark_name: string;
  location?: string;
  latitude?: number;
  longitude?: number;
  confidence?: number;
  added_at: string;
}

class APIClient {
  private baseURL: string;
  private token: string | null = null;

  constructor() {
    this.baseURL = API_BASE_URL;
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('token');
    }
  }

  setToken(token: string) {
    this.token = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('token', token);
      localStorage.setItem('last_login_time', Date.now().toString());
    }
  }

  clearToken() {
    this.token = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token');
      localStorage.removeItem('last_login_time');
      localStorage.removeItem('user');
    }
  }

  private async request(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<any> {
    const headers: Record<string, string> = {
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'API request failed');
    }

    return data;
  }

  async health() {
    return this.request('/health');
  }

  async signup(email: string, password: string, username: string) {
    const response = await this.request('/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ email, password, username }),
    });
    
    if (response.access_token) {
      this.setToken(response.access_token);
    }
    
    return response;
  }

  async login(email: string, password: string) {
    const response = await this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });

    if (response.access_token) {
      this.setToken(response.access_token);
    }

    return response;
  }

  async predict(imageFile: File): Promise<PredictResponse> {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${this.baseURL}/predict`, {
      method: 'POST',
      headers: this.token ? { Authorization: `Bearer ${this.token}` } : {},
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Prediction failed');
    }

    return data;
  }

  async getRecommendations(landmarkName: string) {
    return this.request(`/recommend?landmark_name=${encodeURIComponent(landmarkName)}`);
  }

  async addToItinerary(
    landmarkName: string,
    location?: string,
    latitude?: number,
    longitude?: number,
    confidence?: number
  ) {
    return this.request('/itinerary/add', {
      method: 'POST',
      body: JSON.stringify({
        landmark_name: landmarkName,
        location,
        latitude,
        longitude,
        confidence,
      }),
    });
  }

  async getItinerary(): Promise<{ success: boolean; itinerary: ItineraryItem[] }> {
    return this.request('/itinerary/list');
  }

  async deleteFromItinerary(itineraryId: string) {
    return this.request(`/itinerary/${itineraryId}`, {
      method: 'DELETE',
    });
  }
}

export const apiClient = new APIClient();

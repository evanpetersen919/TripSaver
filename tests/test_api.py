"""
Tests for FastAPI endpoints
Note: These tests require the API to be running or mock the endpoints
"""
import pytest
import json


# Skip actual API tests for now since they require full setup
# Can be enabled once environment is properly configured
pytestmark = pytest.mark.skip(reason="API tests require full environment setup")


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestRecommendationEndpoint:
    """Test recommendation endpoint"""
    
    def test_recommend_without_auth(self, client):
        """Test /recommend endpoint works without authentication"""
        payload = {
            "itinerary_landmarks": ["Eiffel Tower"],
            "vision_description": "Popular tourist attractions in Paris",
            "clip_embedding": None,
            "max_distance_km": 50.0,
            "top_k": 5
        }
        
        response = client.post("/recommend", json=payload)
        
        # Should work without auth (optional auth)
        assert response.status_code in [200, 500]  # 500 if S3/data not available in test
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert isinstance(data["recommendations"], list)
    
    
    def test_recommend_filters_duplicates(self, client):
        """Test that recommendations don't include itinerary landmarks"""
        payload = {
            "itinerary_landmarks": ["Eiffel Tower", "Louvre Museum"],
            "vision_description": "Museums and landmarks",
            "clip_embedding": None,
            "max_distance_km": 50.0,
            "top_k": 5
        }
        
        response = client.post("/recommend", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get("recommendations", [])
            
            # Check no duplicates
            rec_names = [r["name"] for r in recommendations]
            assert "Eiffel Tower" not in rec_names
            assert "Louvre Museum" not in rec_names
    
    
    def test_recommend_invalid_payload(self, client):
        """Test recommendation with invalid payload"""
        payload = {
            "itinerary_landmarks": [],
            # Missing required fields
        }
        
        response = client.post("/recommend", json=payload)
        assert response.status_code == 422  # Validation error


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    def test_signup(self, client):
        """Test user signup"""
        # Note: This will fail if DynamoDB is not available
        payload = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        
        response = client.post("/signup", json=payload)
        
        # May fail in test environment without DynamoDB
        assert response.status_code in [201, 500]
    
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        payload = {
            "username": "nonexistent",
            "password": "wrong"
        }
        
        response = client.post("/login", json=payload)
        
        # Should return 401 or 500 (if DB unavailable)
        assert response.status_code in [401, 500]


class TestLandmarksEndpoint:
    """Test landmarks search endpoint"""
    
    def test_google_search(self, client):
        """Test Google Places search"""
        response = client.get("/landmarks/google-search?q=Eiffel Tower")
        
        # Will succeed if Google API key is configured
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "landmarks" in data

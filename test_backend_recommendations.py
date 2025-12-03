"""
Test script to get auth token and test real ML backend recommendations
"""
import requests
import json

BASE_URL = "https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod"

def test_backend():
    print("=" * 60)
    print("Testing CV Location Backend - Recommendations")
    print("=" * 60)
    
    # Step 1: Create test account or login
    print("\n1. Creating test account...")
    email = "test@example.com"
    password = "TestPassword123!"
    
    signup_response = requests.post(
        f"{BASE_URL}/auth/signup",
        json={
            "email": email,
            "username": "testuser",
            "password": password
        }
    )
    
    token = None
    if signup_response.status_code == 201:
        print("✅ Account created successfully!")
        data = signup_response.json()
        token = data['access_token']
        print(f"   Token: {token[:50]}...")
    else:
        print(f"⚠️  Signup failed (might already exist): {signup_response.status_code}")
        print("   Trying login instead...")
        
        # Try login
        login_response = requests.post(
            f"{BASE_URL}/auth/login",
            json={
                "email": email,
                "password": password
            }
        )
        
        if login_response.status_code == 200:
            print("✅ Login successful!")
            data = login_response.json()
            token = data['access_token']
            print(f"   Token: {token[:50]}...")
        else:
            print(f"❌ Login failed: {login_response.status_code}")
            print(f"   Response: {login_response.text}")
            return
    
    # Step 2: Test recommendations endpoint
    print("\n2. Testing /recommend endpoint with real ML backend...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test with Tokyo landmarks
    recommend_data = {
        "itinerary_landmarks": ["Tokyo Tower"],
        "llava_description": "Popular tourist attractions and landmarks near Tokyo",
        "clip_embedding": None,
        "max_distance_km": 50.0,
        "top_k": 5
    }
    
    print(f"   Request: {json.dumps(recommend_data, indent=2)}")
    
    recommend_response = requests.post(
        f"{BASE_URL}/recommend",
        headers=headers,
        json=recommend_data
    )
    
    print(f"\n   Response Status: {recommend_response.status_code}")
    
    if recommend_response.status_code == 200:
        print("✅ Recommendations received!")
        data = recommend_response.json()
        print(f"\n   Search Mode: {data.get('search_mode')}")
        print(f"   Number of recommendations: {len(data.get('recommendations', []))}")
        print("\n   Recommendations:")
        for i, rec in enumerate(data.get('recommendations', [])[:5], 1):
            print(f"   {i}. {rec.get('name')}")
            print(f"      Location: ({rec.get('latitude')}, {rec.get('longitude')})")
            print(f"      Final Score: {rec.get('final_score', 0):.3f}")
            if rec.get('distance_km'):
                print(f"      Distance: {rec.get('distance_km'):.2f} km")
            print()
    else:
        print(f"❌ Recommendations failed: {recommend_response.status_code}")
        print(f"   Response: {recommend_response.text}")
    
    print("\n" + "=" * 60)
    print("To use this token in frontend, add to localStorage:")
    print(f"localStorage.setItem('auth_token', '{token}');")
    print("=" * 60)

if __name__ == "__main__":
    test_backend()

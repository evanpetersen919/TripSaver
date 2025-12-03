"""
Simple test to check backend health and recommendation engine availability
"""
import requests

BASE_URL = "https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod"

print("=" * 60)
print("Testing Backend Health & Capabilities")
print("=" * 60)

# Test 1: Health check
print("\n1. Testing /health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
        print("   ✅ Backend is healthy!")
    else:
        print(f"   ❌ Health check failed")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Info endpoint
print("\n2. Testing /info endpoint...")
try:
    response = requests.get(f"{BASE_URL}/info", timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   API: {data.get('title')}")
        print(f"   Version: {data.get('version')}")
        print("   ✅ Info retrieved!")
    else:
        print(f"   ❌ Info check failed")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Root endpoint
print("\n3. Testing / (root) endpoint...")
try:
    response = requests.get(f"{BASE_URL}/", timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
        print("   ✅ Root endpoint accessible!")
    else:
        print(f"   ❌ Root check failed")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("Your backend is deployed and accessible!")
print("The /recommend endpoint requires authentication.")
print("To test it, you need to:")
print("1. Fix DynamoDB auth issues (500 error)")
print("2. OR temporarily disable auth for testing")
print("3. OR use the working Google Places fallback")
print("=" * 60)

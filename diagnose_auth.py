"""
Diagnostic test for DynamoDB auth issue
"""
import requests
import json

BASE_URL = "https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod"

print("=" * 60)
print("Diagnosing DynamoDB Auth Issue")
print("=" * 60)

# Test signup with detailed error capture
print("\n1. Testing signup with full error details...")
email = "diagnostic@test.com"
password = "TestPass123!"
username = "diagnosticuser"

signup_payload = {
    "email": email,
    "username": username,
    "password": password
}

print(f"   Payload: {json.dumps(signup_payload, indent=2)}")

try:
    response = requests.post(
        f"{BASE_URL}/auth/signup",
        json=signup_payload,
        timeout=30
    )
    
    print(f"\n   Response Status: {response.status_code}")
    print(f"   Response Headers: {dict(response.headers)}")
    
    try:
        response_data = response.json()
        print(f"   Response Body: {json.dumps(response_data, indent=2)}")
    except:
        print(f"   Response Text: {response.text}")
    
    if response.status_code == 500:
        print("\n   ❌ 500 Internal Server Error - Possible causes:")
        print("   1. Lambda doesn't have DynamoDB permissions")
        print("   2. GSI1 or GSI2 indexes not created")
        print("   3. Environment variables not set")
        print("   4. bcrypt/cryptography library missing in Lambda")
        
except Exception as e:
    print(f"   ❌ Request failed: {e}")

print("\n" + "=" * 60)
print("ACTION ITEMS:")
print("1. Check CloudWatch Logs for Lambda function")
print("2. Verify Lambda IAM role has dynamodb:Query, PutItem")
print("3. Verify GSI1 and GSI2 indexes exist on table")
print("4. Check if bcrypt is in Lambda layer")
print("=" * 60)

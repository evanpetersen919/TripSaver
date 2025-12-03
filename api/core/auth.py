"""
User Authentication System
===========================

JWT-based authentication for AWS Lambda + DynamoDB.
Handles signup, login, token validation, password reset.

Author: Evan Petersen
Date: November 2025
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import boto3
from botocore.exceptions import ClientError

# DynamoDB client
dynamodb = boto3.resource('dynamodb')
TABLE_NAME = os.getenv('DYNAMODB_TABLE', 'cv-location-app')
table = dynamodb.Table(TABLE_NAME)

# JWT configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24


# ============================================================================
# PASSWORD HASHING
# ============================================================================

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify password against hashed version.
    
    Args:
        password: Plain text password
        hashed_password: Bcrypt hashed password
        
    Returns:
        True if password matches
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


# ============================================================================
# JWT TOKEN MANAGEMENT
# ============================================================================

def create_access_token(user_id: str, email: str) -> str:
    """
    Create JWT access token.
    
    Args:
        user_id: User ID
        email: User email
        
    Returns:
        JWT token string
    """
    expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': expiration,
        'iat': datetime.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload or None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


def get_user_from_token(token: str) -> Optional[str]:
    """
    Extract user_id from JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        User ID or None if invalid
    """
    payload = decode_access_token(token)
    if payload:
        return payload.get('user_id')
    return None


# ============================================================================
# USER SIGNUP
# ============================================================================

def signup(email: str, username: str, password: str, 
           first_name: Optional[str] = None,
           last_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create new user account.
    
    Args:
        email: User email (unique)
        username: Username (unique)
        password: Plain text password (will be hashed)
        first_name: Optional first name
        last_name: Optional last name
        
    Returns:
        Dict with success status and user data or error message
    """
    # Generate user ID
    user_id = secrets.token_urlsafe(16)
    
    # Check if email already exists
    try:
        response = table.query(
            IndexName='GSI1',
            KeyConditionExpression='GSI1_PK = :email',
            ExpressionAttributeValues={
                ':email': f'USER#EMAIL#{email}'
            }
        )
        
        if response.get('Items'):
            return {'success': False, 'error': 'Email already registered'}
    
    except ClientError as e:
        return {'success': False, 'error': f'Database error: {str(e)}'}
    
    # Check if username already exists
    try:
        response = table.query(
            IndexName='GSI2',
            KeyConditionExpression='GSI2_PK = :username',
            ExpressionAttributeValues={
                ':username': f'USER#USERNAME#{username}'
            }
        )
        
        if response.get('Items'):
            return {'success': False, 'error': 'Username already taken'}
    
    except ClientError as e:
        return {'success': False, 'error': f'Database error: {str(e)}'}
    
    # Hash password
    hashed_pw = hash_password(password)
    
    # Create user item
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    user_item = {
        'PK': f'USER#{user_id}',
        'SK': 'PROFILE',
        'GSI1_PK': f'USER#EMAIL#{email}',
        'GSI1_SK': f'USER#EMAIL#{email}',
        'GSI2_PK': f'USER#USERNAME#{username}',
        'GSI2_SK': f'USER#USERNAME#{username}',
        'entity_type': 'User',
        'user_id': user_id,
        'email': email,
        'username': username,
        'hashed_password': hashed_pw,
        'first_name': first_name or '',
        'last_name': last_name or '',
        'is_active': True,
        'is_verified': False,
        'created_at': timestamp,
        'updated_at': timestamp
    }
    
    # Save to DynamoDB
    try:
        table.put_item(Item=user_item)
    except ClientError as e:
        return {'success': False, 'error': f'Failed to create user: {str(e)}'}
    
    # Create access token
    token = create_access_token(user_id, email)
    
    # Return success (exclude hashed_password)
    user_data = {k: v for k, v in user_item.items() 
                 if k not in ['hashed_password', 'PK', 'SK', 'GSI1_PK', 'GSI1_SK', 'GSI2_PK', 'GSI2_SK']}
    
    return {
        'success': True,
        'user': user_data,
        'token': token
    }


# ============================================================================
# USER LOGIN
# ============================================================================

def login(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate user and return JWT token.
    
    Args:
        email: User email
        password: Plain text password
        
    Returns:
        Dict with success status and token or error message
    """
    # Get user by email
    try:
        response = table.query(
            IndexName='GSI1',
            KeyConditionExpression='GSI1_PK = :email',
            ExpressionAttributeValues={
                ':email': f'USER#EMAIL#{email}'
            }
        )
        
        items = response.get('Items', [])
        if not items:
            return {'success': False, 'error': 'Invalid email or password'}
        
        user = items[0]
        
    except ClientError as e:
        return {'success': False, 'error': f'Database error: {str(e)}'}
    
    # Verify password
    if not verify_password(password, user['hashed_password']):
        return {'success': False, 'error': 'Invalid email or password'}
    
    # Check if account is active
    if not user.get('is_active', True):
        return {'success': False, 'error': 'Account is deactivated'}
    
    # Update last login timestamp
    user_id = user['user_id']
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    try:
        table.update_item(
            Key={'PK': f'USER#{user_id}', 'SK': 'PROFILE'},
            UpdateExpression='SET last_login = :timestamp',
            ExpressionAttributeValues={':timestamp': timestamp}
        )
    except ClientError:
        pass  # Non-critical, continue even if update fails
    
    # Create access token
    token = create_access_token(user_id, email)
    
    # Return user data (exclude sensitive info)
    user_data = {
        'user_id': user_id,
        'email': user['email'],
        'username': user['username'],
        'first_name': user.get('first_name', ''),
        'last_name': user.get('last_name', ''),
        'is_verified': user.get('is_verified', False),
        'created_at': user.get('created_at')
    }
    
    return {
        'success': True,
        'user': user_data,
        'token': token
    }


# ============================================================================
# GET USER INFO
# ============================================================================

def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user profile by ID.
    
    Args:
        user_id: User ID
        
    Returns:
        User data dict or None if not found
    """
    try:
        response = table.get_item(
            Key={'PK': f'USER#{user_id}', 'SK': 'PROFILE'}
        )
        
        if 'Item' not in response:
            return None
        
        user = response['Item']
        
        # Remove sensitive data
        return {
            'user_id': user['user_id'],
            'email': user['email'],
            'username': user['username'],
            'first_name': user.get('first_name', ''),
            'last_name': user.get('last_name', ''),
            'is_active': user.get('is_active', True),
            'is_verified': user.get('is_verified', False),
            'created_at': user.get('created_at'),
            'last_login': user.get('last_login')
        }
    
    except ClientError:
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    try:
        response = table.query(
            IndexName='GSI1',
            KeyConditionExpression='GSI1_PK = :email',
            ExpressionAttributeValues={
                ':email': f'USER#EMAIL#{email}'
            }
        )
        
        items = response.get('Items', [])
        if not items:
            return None
        
        user = items[0]
        return get_user(user['user_id'])
    
    except ClientError:
        return None


# ============================================================================
# TOKEN VALIDATION (Middleware)
# ============================================================================

def require_auth(token: str) -> Dict[str, Any]:
    """
    Validate JWT token and return user info.
    Use this as middleware for protected routes.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        Dict with success status and user data or error
    """
    if not token:
        return {'success': False, 'error': 'No token provided'}
    
    # Remove "Bearer " prefix if present
    if token.startswith('Bearer '):
        token = token[7:]
    
    # Decode token
    payload = decode_access_token(token)
    if not payload:
        return {'success': False, 'error': 'Invalid or expired token'}
    
    # Get user from database
    user_id = payload.get('user_id')
    user = get_user(user_id)
    
    if not user:
        return {'success': False, 'error': 'User not found'}
    
    if not user.get('is_active', True):
        return {'success': False, 'error': 'Account is deactivated'}
    
    return {'success': True, 'user': user}


# ============================================================================
# PASSWORD RESET (Basic Implementation)
# ============================================================================

def request_password_reset(email: str) -> Dict[str, Any]:
    """
    Generate password reset token.
    In production, send this token via email.
    
    Args:
        email: User email
        
    Returns:
        Dict with success status and reset token
    """
    user = get_user_by_email(email)
    
    if not user:
        # Don't reveal if email exists (security best practice)
        return {'success': True, 'message': 'If email exists, reset link will be sent'}
    
    # Generate reset token (valid for 1 hour)
    expiration = datetime.utcnow() + timedelta(hours=1)
    reset_token = jwt.encode(
        {'user_id': user['user_id'], 'purpose': 'reset', 'exp': expiration},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )
    
    # TODO: Send email with reset_token
    # For now, return token directly (in production, send via email)
    
    return {
        'success': True,
        'message': 'Password reset token generated',
        'reset_token': reset_token  # Remove this in production!
    }


def reset_password(reset_token: str, new_password: str) -> Dict[str, Any]:
    """
    Reset password using reset token.
    
    Args:
        reset_token: JWT reset token
        new_password: New plain text password
        
    Returns:
        Dict with success status
    """
    try:
        payload = jwt.decode(reset_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        if payload.get('purpose') != 'reset':
            return {'success': False, 'error': 'Invalid reset token'}
        
        user_id = payload.get('user_id')
        
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return {'success': False, 'error': 'Reset token expired or invalid'}
    
    # Hash new password
    hashed_pw = hash_password(new_password)
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Update password in database
    try:
        table.update_item(
            Key={'PK': f'USER#{user_id}', 'SK': 'PROFILE'},
            UpdateExpression='SET hashed_password = :pw, updated_at = :ts',
            ExpressionAttributeValues={
                ':pw': hashed_pw,
                ':ts': timestamp
            }
        )
    except ClientError as e:
        return {'success': False, 'error': f'Failed to update password: {str(e)}'}
    
    return {'success': True, 'message': 'Password updated successfully'}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("AUTHENTICATION SYSTEM TEST")
    print("=" * 80)
    
    # Test signup
    print("\n1. Testing signup...")
    result = signup(
        email='test@example.com',
        username='testuser',
        password='SecurePassword123!',
        first_name='Test',
        last_name='User'
    )
    
    if result['success']:
        print(f"✓ Signup successful!")
        print(f"  User ID: {result['user']['user_id']}")
        print(f"  Token: {result['token'][:20]}...")
        token = result['token']
    else:
        print(f"✗ Signup failed: {result['error']}")
    
    # Test login
    print("\n2. Testing login...")
    result = login('test@example.com', 'SecurePassword123!')
    
    if result['success']:
        print(f"✓ Login successful!")
        print(f"  Username: {result['user']['username']}")
        print(f"  Token: {result['token'][:20]}...")
    else:
        print(f"✗ Login failed: {result['error']}")
    
    # Test token validation
    print("\n3. Testing token validation...")
    result = require_auth(token)
    
    if result['success']:
        print(f"✓ Token valid!")
        print(f"  User: {result['user']['username']}")
    else:
        print(f"✗ Token invalid: {result['error']}")
    
    print("\n" + "=" * 80)
    print("✓ Authentication system ready!")
    print("=" * 80)

"""
FastAPI Backend for CV Location Classifier
===========================================

REST API for:
- User authentication (signup, login, password reset)
- Image upload and landmark prediction
- Travel recommendations
- User itinerary management

Author: Evan Petersen
Date: January 2025
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import base64
import io
from PIL import Image
import numpy as np
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import os
from dotenv import load_dotenv

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.auth import (
    signup, login, require_auth, request_password_reset, 
    reset_password, decode_access_token
)
from models.huggingface_client import HuggingFaceClient
# LandmarkDetector now called via HF Space API
from models.clip_embedder import ClipEmbedder
from core.recommendation_engine import RecommendationEngine
from core.config import config

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="CV Location Classifier API",
    description="AI-powered landmark detection and travel recommendations",
    version="1.0.0"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# DynamoDB client
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)
table = dynamodb.Table(os.getenv('DYNAMODB_TABLE', 'cv-location-app'))


# ============================================================================
# GLOBAL MODEL INSTANCES (lazy loaded for Lambda cold start optimization)
# ============================================================================

huggingface_client = None
# landmark_detector now via HF Space (no global state)
clip_embedder = None
recommendation_engine = None


def get_huggingface_client():
    """Lazy load Hugging Face client"""
    global huggingface_client
    if huggingface_client is None:
        hf_token = os.getenv('HUGGINGFACE_API_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_TOKEN not set in environment")
        huggingface_client = HuggingFaceClient(api_token=hf_token)
    return huggingface_client


def get_landmark_detector():
    """Call Hugging Face Space for landmark detection"""
    # Use HF Space instead of local model
    import requests
    
    HF_SPACE_URL = "https://evanpetersen919-cv-location-classifier.hf.space/predict"
    
    class HFSpaceLandmarkDetector:
        """Wrapper to call HF Space API"""
        
        def predict(self, image: Image.Image, top_k: int = 5):
            """Send image to HF Space and get predictions"""
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Call HF Space
            files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
            response = requests.post(HF_SPACE_URL, files=files, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get('success'):
                raise Exception(f"HF Space error: {result.get('error', 'Unknown error')}")
            
            # Convert to expected format (class_id -> landmark name)
            # For now, use class_id as landmark name (you'll need a mapping file)
            predictions = []
            for pred in result['predictions'][:top_k]:
                predictions.append({
                    'landmark': f"landmark_{pred['class_id']}",  # TODO: Add proper name mapping
                    'confidence': pred['confidence'],
                    'class_idx': pred['class_id']
                })
            
            return predictions
    
    return HFSpaceLandmarkDetector()


def get_clip_embedder():
    """Lazy load CLIP embedder"""
    global clip_embedder
    if clip_embedder is None:
        clip_embedder = ClipEmbedder(device='cpu')
    return clip_embedder


def get_recommendation_engine():
    """Lazy load recommendation engine"""
    global recommendation_engine
    if recommendation_engine is None:
        landmarks_path = os.getenv('LANDMARKS_DATABASE_PATH')
        if not landmarks_path or not Path(landmarks_path).exists():
            raise ValueError(f"Landmarks database not found: {landmarks_path}")
        recommendation_engine = RecommendationEngine(landmarks_path=landmarks_path)
    return recommendation_engine


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SignupRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=30)
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def validate_username(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    email: EmailStr
    reset_token: str
    new_password: str = Field(..., min_length=8)


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    llava_description: Optional[str] = None
    confidence_level: str
    recommendation_strategy: str
    alternatives: Optional[List[str]] = None


class RecommendationRequest(BaseModel):
    itinerary_landmarks: List[str] = Field(default_factory=list)
    llava_description: str
    clip_embedding: Optional[List[float]] = None
    max_distance_km: Optional[float] = 50.0
    top_k: int = Field(default=5, ge=1, le=20)


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    search_mode: str  # 'proximity' or 'global'


class ItineraryAddRequest(BaseModel):
    landmark_name: str
    landmark_id: int
    latitude: float
    longitude: float
    country: str


class FeedbackRequest(BaseModel):
    prediction_id: str
    was_correct: bool
    correct_landmark: Optional[str] = None
    user_comment: Optional[str] = None


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup_endpoint(request: SignupRequest):
    """
    Register a new user.
    
    Creates a user account with hashed password and returns JWT token.
    """
    try:
        result = signup(
            email=request.email,
            username=request.username,
            password=request.password
        )
        return {
            "message": "User created successfully",
            "user_id": result['user_id'],
            "access_token": result['access_token'],
            "token_type": "bearer"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/auth/login")
async def login_endpoint(request: LoginRequest):
    """
    Authenticate user and return JWT token.
    """
    try:
        result = login(
            email=request.email,
            password=request.password
        )
        return {
            "access_token": result['access_token'],
            "token_type": "bearer",
            "user_id": result['user_id'],
            "username": result['username']
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/auth/password-reset/request")
async def request_password_reset_endpoint(request: PasswordResetRequest):
    """
    Request a password reset token (sent to email in production).
    """
    try:
        result = request_password_reset(email=request.email)
        # In production, send email here
        # For now, return token directly (REMOVE IN PRODUCTION)
        return {
            "message": "Password reset token generated",
            "reset_token": result['reset_token']  # REMOVE IN PRODUCTION
        }
    except ValueError as e:
        # Don't reveal whether email exists
        return {"message": "If the email exists, a reset token has been sent"}


@app.post("/auth/password-reset/confirm")
async def reset_password_endpoint(request: PasswordResetConfirm):
    """
    Reset password using reset token.
    """
    try:
        reset_password(
            email=request.email,
            reset_token=request.reset_token,
            new_password=request.new_password
        )
        return {"message": "Password reset successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# USER ENDPOINTS
# ============================================================================

@app.get("/user/profile")
async def get_user_profile(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get user profile information.
    """
    user_id = require_auth(credentials.credentials)
    
    try:
        response = table.get_item(
            Key={'PK': f'USER#{user_id}', 'SK': 'PROFILE'}
        )
        
        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = response['Item']
        return {
            "user_id": user_id,
            "email": user['email'],
            "username": user['username'],
            "created_at": user['created_at'],
            "itinerary_count": user.get('itinerary_count', 0),
            "prediction_count": user.get('prediction_count', 0)
        }
    except ClientError as e:
        raise HTTPException(status_code=500, detail="Database error")


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

def decode_image_from_base64(image_data: str) -> Image.Image:
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict_landmark(
    image: str = None,  # Base64 encoded image
    file: UploadFile = File(None),  # Or file upload
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Predict landmark from uploaded image.
    
    Uses:
    1. EfficientNet landmark detector (500 classes)
    2. LLaVA via Hugging Face API (scene understanding)
    3. CLIP embedder (visual similarity)
    
    Returns predictions with confidence levels and recommendation strategy.
    """
    user_id = require_auth(credentials.credentials)
    
    # Load image
    if file:
        image_data = await file.read()
        image_pil = Image.open(io.BytesIO(image_data))
    elif image:
        image_pil = decode_image_from_base64(image)
    else:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Ensure RGB
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    try:
        # 1. Run landmark detector
        detector = get_landmark_detector()
        predictions = detector.predict(image_pil, top_k=5)
        
        # 2. Run LLaVA via Hugging Face (for scene understanding)
        hf_client = get_huggingface_client()
        llava_result = hf_client.analyze_location(image_pil)
        
        llava_description = llava_result.get('description', '') if llava_result['success'] else None
        
        # 3. Run CLIP embedder (for visual similarity)
        clip = get_clip_embedder()
        clip_embedding = clip.encode_image(image_pil)
        
        # Determine recommendation strategy
        top_confidence = predictions[0]['confidence']
        second_confidence = predictions[1]['confidence'] if len(predictions) > 1 else 0
        confidence_gap = top_confidence - second_confidence
        
        # Calibrated confidence levels
        if top_confidence >= 0.95 and confidence_gap >= 0.3:
            confidence_level = 'high'
            strategy = 'landmark'
            alternatives = None
        elif top_confidence >= 0.70 and confidence_gap >= 0.10:
            confidence_level = 'medium'
            strategy = 'landmark_options'
            alternatives = [p['landmark'] for p in predictions[:3]]
        else:
            confidence_level = 'low'
            strategy = 'scene'
            alternatives = [p['landmark'] for p in predictions[:5]]
        
        # Save prediction to DynamoDB
        prediction_id = f"{user_id}#{int(datetime.utcnow().timestamp() * 1000)}"
        
        table.put_item(Item={
            'PK': f'USER#{user_id}',
            'SK': f'PREDICTION#{prediction_id}',
            'prediction_id': prediction_id,
            'top_prediction': predictions[0]['landmark'],
            'top_confidence': str(predictions[0]['confidence']),
            'all_predictions': predictions,
            'llava_description': llava_description,
            'confidence_level': confidence_level,
            'timestamp': datetime.utcnow().isoformat(),
            'GSI1_PK': f'PREDICTION#{prediction_id}',
            'GSI1_SK': 'METADATA'
        })
        
        return PredictResponse(
            predictions=predictions,
            llava_description=llava_description,
            confidence_level=confidence_level,
            recommendation_strategy=strategy,
            alternatives=alternatives
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# RECOMMENDATION ENDPOINTS
# ============================================================================

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get landmark recommendations based on itinerary and image features.
    
    Two modes:
    1. Proximity search: If itinerary provided, find nearby landmarks
    2. Global search: If no itinerary, search entire database by similarity
    """
    user_id = require_auth(credentials.credentials)
    
    try:
        engine = get_recommendation_engine()
        
        # Convert CLIP embedding from list to numpy array
        clip_embedding = None
        if request.clip_embedding:
            clip_embedding = np.array(request.clip_embedding, dtype=np.float32)
        
        # Determine search mode
        is_global_search = not request.itinerary_landmarks or request.max_distance_km is None
        
        if is_global_search:
            # Global content-based search
            results = engine.search_by_description(
                llava_description=request.llava_description,
                clip_embedding=clip_embedding,
                top_k=request.top_k,
                min_similarity=0.3
            )
            search_mode = 'global'
        else:
            # Proximity-based search
            recs = engine.recommend(
                itinerary_landmarks=request.itinerary_landmarks,
                llava_description=request.llava_description,
                max_distance_km=request.max_distance_km,
                top_k=request.top_k,
                clip_embedding=clip_embedding
            )
            
            # Convert Recommendation objects to dicts
            results = [
                {
                    'name': rec.name,
                    'landmark_id': rec.landmark_id,
                    'latitude': rec.latitude,
                    'longitude': rec.longitude,
                    'distance_km': rec.distance_km,
                    'similarity_score': rec.similarity_score,
                    'final_score': rec.final_score,
                    'country': rec.country,
                    'description': rec.description,
                    'closest_itinerary_item': rec.closest_itinerary_item
                }
                for rec in recs
            ]
            search_mode = 'proximity'
        
        return RecommendationResponse(
            recommendations=results,
            search_mode=search_mode
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


# ============================================================================
# ITINERARY ENDPOINTS
# ============================================================================

@app.post("/itinerary/add")
async def add_to_itinerary(
    request: ItineraryAddRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Add a landmark to user's itinerary.
    """
    user_id = require_auth(credentials.credentials)
    
    try:
        itinerary_id = f"{user_id}#{int(datetime.utcnow().timestamp() * 1000)}"
        
        table.put_item(Item={
            'PK': f'USER#{user_id}',
            'SK': f'ITINERARY#{itinerary_id}',
            'itinerary_id': itinerary_id,
            'landmark_name': request.landmark_name,
            'landmark_id': request.landmark_id,
            'latitude': str(request.latitude),
            'longitude': str(request.longitude),
            'country': request.country,
            'added_at': datetime.utcnow().isoformat(),
            'visited': False,
            'GSI1_PK': f'ITINERARY#{itinerary_id}',
            'GSI1_SK': 'METADATA'
        })
        
        return {"message": "Added to itinerary", "itinerary_id": itinerary_id}
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/itinerary/list")
async def get_itinerary(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get user's itinerary.
    """
    user_id = require_auth(credentials.credentials)
    
    try:
        response = table.query(
            KeyConditionExpression='PK = :pk AND begins_with(SK, :sk)',
            ExpressionAttributeValues={
                ':pk': f'USER#{user_id}',
                ':sk': 'ITINERARY#'
            }
        )
        
        itinerary = response.get('Items', [])
        
        # Sort by added_at descending
        itinerary.sort(key=lambda x: x['added_at'], reverse=True)
        
        return {"itinerary": itinerary, "count": len(itinerary)}
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail="Database error")


@app.delete("/itinerary/{itinerary_id}")
async def remove_from_itinerary(
    itinerary_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Remove a landmark from itinerary.
    """
    user_id = require_auth(credentials.credentials)
    
    try:
        table.delete_item(
            Key={
                'PK': f'USER#{user_id}',
                'SK': f'ITINERARY#{itinerary_id}'
            }
        )
        
        return {"message": "Removed from itinerary"}
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail="Database error")


# ============================================================================
# FEEDBACK ENDPOINTS
# ============================================================================

@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Submit feedback on a prediction.
    
    Used to improve model accuracy over time.
    """
    user_id = require_auth(credentials.credentials)
    
    try:
        feedback_id = f"{user_id}#{int(datetime.utcnow().timestamp() * 1000)}"
        
        table.put_item(Item={
            'PK': f'USER#{user_id}',
            'SK': f'FEEDBACK#{feedback_id}',
            'feedback_id': feedback_id,
            'prediction_id': request.prediction_id,
            'was_correct': request.was_correct,
            'correct_landmark': request.correct_landmark,
            'user_comment': request.user_comment,
            'timestamp': datetime.utcnow().isoformat(),
            'GSI1_PK': f'FEEDBACK#{feedback_id}',
            'GSI1_SK': 'METADATA'
        })
        
        return {"message": "Feedback submitted", "feedback_id": feedback_id}
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail="Database error")


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/info")
async def get_info():
    """
    Get API information and model details.
    """
    try:
        detector = get_landmark_detector()
        engine = get_recommendation_engine()
        
        return {
            "api_version": "1.0.0",
            "models": {
                "landmark_detector": {
                    "num_classes": detector.num_classes,
                    "architecture": "EfficientNet-B3"
                },
                "llava": {
                    "provider": "Hugging Face Inference API",
                    "model": "llava-hf/llava-1.5-7b-hf"
                },
                "clip": {
                    "architecture": "CLIP ViT-B/32"
                },
                "recommendation_engine": {
                    "database_size": len(engine.landmarks),
                    "embedder": "all-MiniLM-L6-v2"
                }
            }
        }
    except Exception as e:
        return {
            "api_version": "1.0.0",
            "models": "Not loaded (cold start)"
        }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "CV Location Classifier API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "auth": ["/auth/signup", "/auth/login", "/auth/password-reset/request"],
            "predictions": ["/predict"],
            "recommendations": ["/recommend"],
            "itinerary": ["/itinerary/add", "/itinerary/list", "/itinerary/{id}"],
            "feedback": ["/feedback"],
            "info": ["/health", "/info"]
        }
    }


# ============================================================================
# LAMBDA HANDLER (for AWS deployment)
# ============================================================================

# Import mangum for AWS Lambda
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    # Mangum not installed - running locally
    handler = None


# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("CV LOCATION CLASSIFIER API")
    print("=" * 80)
    print("Starting development server...")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

# TripSaver - AI-Powered Trip Planner
### Full-Stack AI Platform for Global Landmark Recognition

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-Lambda-orange.svg)](https://aws.amazon.com/lambda/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)

A full-stack AI platform that transforms travel photos into landmark discoveries. Upload any image and instantly identify landmarks across 500 classes with visual similarity search spanning 4,248 unique landmarks.

## Key Features

- **Two-Tier Detection System**: EfficientNet-B3 (500 classes) with Google Vision API validation for low-confidence results
- **Visual Similarity Search**: CLIP ViT-B/32 embeddings with FAISS search across 4,248 landmarks
- **Scene Understanding**: Groq Llama 4 Scout 17B for detailed scene descriptions
- **Social Media Ready**: Custom augmentation pipeline for Instagram/TikTok overlays and filters
- **Smart Recommendations**: Finds nearby attractions based on your itinerary and image features
- **Full-Stack Web App**: Next.js frontend deployed on Vercel with real-time image search
- **Serverless Architecture**: AWS Lambda + API Gateway for scalable inference
- **Secure Authentication**: JWT with bcrypt password hashing

## System Architecture

```
┌─────────────────────┐
│   Next.js Frontend  │  (Vercel - Auto Deploy)
│   React 19 + Tailwind│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    API Gateway      │  (AWS - REST)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Lambda Functions   │  (FastAPI + JWT Auth)
└──────┬──────────────┘
       │
       ├──→ HuggingFace Space (CPU)
       │    └─ EfficientNet-B3 (500 classes)
       │
       ├──→ Google Vision API
       │    └─ Validation (confidence < 70%)
       │
       ├──→ CLIP + Groq (Tier 2)
       │    ├─ CLIP ViT-B/32 (4,248 embeddings)
       │    └─ Llama 4 Scout 17B (scene understanding)
       │
       └──→ DynamoDB + FAISS
            ├─ Users, Itineraries, Predictions
            └─ 4,248 landmark vectors
```

## Project Structure

```
cv_pipeline/
├── api/                      # FastAPI REST API
│   └── main.py              # All endpoints (auth, predict, recommend)
├── core/                     # Business logic
│   ├── auth.py              # JWT + bcrypt authentication
│   ├── vision_pipeline.py   # Model orchestration
│   └── recommendation_engine.py  # Travel recommendations
├── models/                   # ML models
│   ├── landmark_detector.py # EfficientNet-B3 (500 classes)
│   ├── clip_embedder.py     # Visual similarity search
│   ├── huggingface_client.py # HuggingFace Space integration
│   └── (removed llava_analyzer.py - now using Groq)
├── aws/                      # AWS configuration
│   ├── dynamodb_schema.json # Single-table design
│   └── README.md            # DynamoDB documentation
├── scripts/                  # Deployment scripts
│   └── build_lambda_layers.py # Package dependencies + models
├── data/                     # Trained models & databases
│   ├── checkpoints/         # EfficientNet weights
│   └── landmarks_unified.json # 50K+ landmark database
├── template.yaml            # AWS SAM Infrastructure as Code
├── samconfig.toml           # SAM deployment config
├── .env.example             # Environment variables template
├── DEPLOYMENT.md            # Step-by-step deployment guide
└── requirements.txt         # Python dependencies
```

## Local Setup

### Prerequisites
- Python 3.11+
- AWS Account
- HuggingFace API token
- AWS CLI + SAM CLI installed

### Run Locally

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with tokens:
# - HUGGINGFACE_API_TOKEN
# - JWT_SECRET
# - GROQ_API_KEY
```

3. **Run API**
```bash
cd api
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Deploy to AWS

1. **Build Lambda layers**
```bash
python scripts/build_lambda_layers.py
```

2. **Deploy with SAM**
```bash
sam build --use-container
sam deploy --guided
```

3. **Get API endpoint**
```bash
aws cloudformation describe-stacks \
  --stack-name cv-location-classifier \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
  --output text
```

## Deployment

- **Frontend**: Vercel (auto-deploy on GitHub push)
- **Backend**: AWS Lambda + API Gateway (deployed via SAM)
- **Model**: HuggingFace Spaces (CPU inference)
- **Database**: DynamoDB + FAISS vector storage

## Authentication Flow

### How User Login Works

1. **User Signup** (`POST /auth/signup`)
```javascript
// Request
{
  "email": "user@example.com",
  "username": "traveler123",
  "password": "SecurePass123!"
}

// Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "uuid-here"
}
```

Backend process:
- Validates email/username uniqueness (DynamoDB query on GSI)
- Hashes password with bcrypt (12 rounds)
- Creates user record in DynamoDB
- Generates JWT token (HS256, 24hr expiration)
- Returns token + user_id

2. **User Login** (`POST /auth/login`)
```javascript
// Request
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}

// Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "uuid-here",
  "username": "traveler123"
}
```

Backend process:
- Looks up user by email (DynamoDB GSI1)
- Verifies password with bcrypt
- Generates new JWT token
- Returns token + user info

3. **Protected Routes** (all other endpoints)
```javascript
// Request headers
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Backend process:
- Extracts JWT from Authorization header
- Verifies signature with JWT_SECRET
- Checks expiration (24 hours)
- Decodes user_id from token payload
- Proceeds with request using authenticated user_id

### JWT Token Structure
```json
{
  "user_id": "uuid-here",
  "exp": 1732838400,  // Unix timestamp (24hr from creation)
  "iat": 1732752000   // Issued at timestamp
}
```

### Password Reset Flow
1. Request reset token: `POST /auth/password-reset/request`
2. Backend generates time-limited token (1 hour expiration)
3. Token stored in DynamoDB with user record
4. User provides token + new password: `POST /auth/password-reset/confirm`
5. Backend validates token, updates password, invalidates token



## API Endpoints

### Authentication
- `POST /auth/signup` - Create account
- `POST /auth/login` - Get JWT token
- `POST /auth/password-reset/request` - Request reset
- `POST /auth/password-reset/confirm` - Reset password

### Predictions
- `POST /predict` - Identify landmark in image
- Returns: predictions, confidence, vision description (via Groq)

### Recommendations
- `POST /recommend` - Get nearby attractions
- Modes: proximity search or global search

### Itinerary Management
- `POST /itinerary/add` - Add landmark
- `GET /itinerary/list` - View itinerary
- `DELETE /itinerary/{id}` - Remove landmark

### User
- `GET /user/profile` - User info

### Info
- `GET /health` - Health check
- `GET /info` - Model details
- `GET /docs` - OpenAPI documentation

## ML Models

### 1. EfficientNet-B3 (Primary Classifier)
- **Dataset**: Google Landmarks Dataset v2 (500 classes)
- **Training**: 20 hours on RTX 4080 (i9-13900K, 64GB DDR5)
- **Augmentation**: RandAugment, MixUp, CutMix + custom social media overlays
- **Input**: 300x300 RGB images
- **Output**: Top-5 predictions with confidence scores
- **Inference**: HuggingFace Spaces (CPU)
- **Preprocessing**: Template matching to remove social media UI elements

### 2. Google Vision API (Validation)
- **Purpose**: Validates EfficientNet predictions when confidence < 70%
- **Integration**: Google Cloud Vision API
- **Usage**: Tier 1 fallback for low-confidence results

### 3. CLIP ViT-B/32 (Visual Similarity)
- **Database**: 4,248 landmark embeddings
- **Search Method**: FAISS cosine similarity
- **Purpose**: Tier 2 fallback when users reject initial predictions
- **Search Time**: <50ms

### 4. Groq Llama 4 Scout 17B (Scene Understanding)
- **Provider**: Groq API (ultra-fast inference)
- **Purpose**: Detailed scene descriptions and context
- **Activation**: Tier 2 fallback alongside CLIP search
- **Model**: meta-llama/llama-4-scout-17b-16e-instruct

## DynamoDB Schema

Single-table design with two GSI indexes:

```
Primary Key: PK (partition), SK (sort)

Entities:
- Users: PK=USER#{id}, SK=PROFILE
- Itineraries: PK=USER#{id}, SK=ITINERARY#{id}
- Predictions: PK=USER#{id}, SK=PREDICTION#{id}
- Feedback: PK=USER#{id}, SK=FEEDBACK#{id}

GSI1: Email lookup (GSI1_PK=EMAIL#{email})
GSI2: Username lookup (GSI2_PK=USERNAME#{username})
```

## Cost Breakdown

### AWS Always Free Tier (Permanent)
- **Lambda**: 1M requests/month + 400K GB-seconds compute
- **API Gateway**: 1M API calls/month
- **DynamoDB**: 25 GB storage + 25 RCU/sec + 25 WCU/sec

### Hugging Face (Free Tier)
- **Inference API**: 1,000 requests/month
- **Rate Limit**: 1 request/second

### Total Monthly Cost: **$0.00**



## Development Commands

### Testing
```bash
pytest tests/ -v
```

### Local API with Hot Reload
```bash
cd api
uvicorn main:app --reload --port 8000
```

### View Logs (Deployed)
```bash
sam logs --stack-name cv-location-classifier --tail
```

### Monitor DynamoDB
```bash
aws dynamodb scan --table-name cv-location-app --max-items 10
```

## Technical Highlights

**Training Pipeline:**
- Custom hardware: i9-13900K, RTX 4080 16GB, 64GB DDR5, Intel 1TB NVMe
- Mixed precision FP16 training with AdamW optimizer
- Transfer learning from ImageNet pretrained weights
- Social media augmentation (Instagram/TikTok overlays at 30% probability)
- MLflow for experiment tracking and model registry

**Detection System:**
- Two-tier architecture: Fast classification (Tier 1) + Fallback analysis (Tier 2)
- Tier 1: EfficientNet-B3 with optional Google Vision validation
- Tier 2: CLIP similarity search + Groq LLM for rejected predictions
- Preprocessing removes social media UI using template matching

**Full-Stack Application:**
- Next.js 16 (React 19) frontend with Tailwind CSS
- FastAPI backend with JWT authentication
- DynamoDB single-table design with GSI indexes
- AWS Lambda serverless deployment
- Vercel auto-deployment for frontend
- GitHub Actions CI/CD pipeline
- CloudWatch monitoring for Lambda performance

## License

All Rights Reserved © 2025 Evan Petersen

## Author

**Evan Petersen**  
[GitHub](https://github.com/evanpetersen919) | [LinkedIn](https://www.linkedin.com/in/evan-petersen-b93037386/)

## Acknowledgments

- Google Landmarks Dataset v2
- Groq for FREE ultra-fast vision model inference
- Hugging Face Spaces for model hosting
- AWS for Always Free Tier
- FastAPI and PyTorch communities

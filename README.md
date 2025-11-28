# CV Location Classifier
### AI-Powered Landmark Detection & Travel Recommendation System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-Lambda-orange.svg)](https://aws.amazon.com/lambda/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A serverless AI travel companion that identifies landmarks in photos and recommends nearby attractions. Deployed entirely on AWS Always Free Tier with zero monthly costs.

## Key Features

- **Landmark Detection**: EfficientNet-B3 trained on 500 landmark classes (~80% accuracy)
- **Scene Understanding**: LLaVA-1.5 via Hugging Face API for natural language descriptions
- **Visual Similarity**: CLIP embeddings for content-based landmark search
- **Smart Recommendations**: Finds nearby attractions based on your itinerary + image features
- **JWT Authentication**: Secure user accounts with bcrypt password hashing
- **DynamoDB Single-Table Design**: Efficient NoSQL data modeling with GSI indexes
- **Serverless Architecture**: AWS Lambda + API Gateway (1M free requests/month)
- **One-Command Deployment**: Infrastructure as Code with AWS SAM

## Architecture

```
┌─────────────────────┐
│    API Gateway      │  (1M calls/month - FREE)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Lambda Function    │  (1M requests/month - FREE)
│   FastAPI + Models  │
└──────┬──────────────┘
       │
       ├──→ DynamoDB (25GB - FREE)
       │    └─ Users, Itineraries, Predictions
       │
       ├──→ Hugging Face API (1K/month - FREE)
       │    └─ LLaVA-1.5-7B inference
       │
       └──→ Local Models (Lambda CPU)
            └─ EfficientNet-B3, CLIP
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
│   ├── huggingface_client.py # LLaVA API integration
│   └── llava_analyzer.py    # Local LLaVA (deprecated)
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

## Quick Start

### Prerequisites
- Python 3.11+
- AWS Account (free tier)
- Hugging Face API token ([get here](https://huggingface.co/settings/tokens))
- AWS CLI + SAM CLI installed

### Local Development

1. **Clone repository**
```bash
git clone https://github.com/evanpetersen919/CV-Location-Classifier.git
cd CV-Location-Classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your tokens:
# - HUGGINGFACE_API_TOKEN
# - JWT_SECRET
```

4. **Run API locally**
```bash
cd api
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### AWS Deployment

1. **Build Lambda layers**
```bash
python scripts/build_lambda_layers.py
```

2. **Deploy with SAM**
```bash
sam build --use-container
sam deploy --guided
```

3. **Get your API endpoint**
```bash
aws cloudformation describe-stacks \
  --stack-name cv-location-classifier \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
  --output text
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide.

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
- Returns: predictions, confidence, LLaVA description

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

### 1. Landmark Detector (EfficientNet-B3)
- **Dataset**: 500 landmark classes
- **Accuracy**: ~80% on validation set
- **Input**: 300x300 RGB images
- **Output**: Top-5 predictions with confidence scores
- **Inference Time**: ~200ms on Lambda CPU

### 2. LLaVA (Vision-Language Model)
- **Model**: llava-hf/llava-1.5-7b-hf
- **Provider**: Hugging Face Inference API
- **Purpose**: Natural language scene descriptions
- **Rate Limit**: 1 request/second (free tier)
- **Monthly Limit**: 1,000 requests (free tier)

### 3. CLIP (Visual Similarity)
- **Architecture**: ViT-B/32
- **Purpose**: Find visually similar landmarks
- **Database**: 50K+ landmark embeddings
- **Search Time**: <50ms (cosine similarity)

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

## Development

### Run Tests
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

## Technical Implementation

**Completed:**
- Landmark detection (EfficientNet-B3, 500 classes)
- Scene understanding (LLaVA-1.5 via Hugging Face API)
- Visual similarity search (CLIP embeddings)
- User authentication (JWT + bcrypt)
- DynamoDB single-table design with GSI indexes
- FastAPI REST API with 12 endpoints
- AWS Lambda serverless deployment
- Infrastructure as Code with AWS SAM
- Comprehensive deployment documentation

## License

MIT License

## Author

**Evan Petersen**  
[GitHub](https://github.com/evanpetersen919) | [LinkedIn](https://www.linkedin.com/in/evan-petersen-b93037386/)

## Acknowledgments

- Google Landmarks Dataset v2
- Hugging Face for LLaVA API
- AWS for Always Free Tier
- FastAPI and PyTorch communities

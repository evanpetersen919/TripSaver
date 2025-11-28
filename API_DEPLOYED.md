# CV Location Classifier - Deployed Backend

## API Endpoint
**Base URL:** `https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod/`

## Deployment Summary
- ✅ Lambda Function: `cv-location-classifier` 
- ✅ DynamoDB Table: `cv-location-app`
- ✅ API Gateway: `eh5scbzco7`
- ✅ Region: `us-east-1`

## Architecture
- **Frontend** → **AWS API Gateway** → **AWS Lambda** (FastAPI) → **HuggingFace Space** (Model Inference)
- **Database**: DynamoDB
- **Model**: EfficientNet-B3 (500 classes) hosted on HF Spaces

## Available Endpoints
- `GET /` - Root
- `GET /health` - Health check
- `GET /info` - API info
- `POST /auth/signup` - User signup
- `POST /auth/login` - User login  
- `POST /predict` - Image prediction
- `POST /recommend` - Get recommendations
- `GET /itinerary/list` - List itineraries
- `POST /itinerary/add` - Add to itinerary
- `DELETE /itinerary/{id}` - Delete itinerary

## Next Steps
1. Test API endpoints
2. Build frontend (React/Next.js)
3. Connect frontend to API Gateway

## Cost
**100% Free Forever** - Stays within AWS Free Tier limits

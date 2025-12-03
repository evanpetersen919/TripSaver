# TODO: Fix HuggingFace Inference API

## Issue
HuggingFace's old inference API endpoint is deprecated (returns 410).
- Old endpoint: `https://api-inference.huggingface.co/pipeline/...` (deprecated)
- Need to find new serverless inference endpoint

## Current Status
- System working with **keyword-based similarity** (good for proximity search)
- Pre-computed embeddings ready at `data/embeddings/landmark_text_embeddings.npy` (15,873 landmarks, 23MB)
- Metadata at `data/embeddings/landmark_text_metadata.json`
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

## What to Fix Tomorrow
1. Find HuggingFace's new serverless inference API endpoint
   - Check HF docs: https://huggingface.co/docs/api-inference
   - Or use dedicated inference endpoints (might require paid tier)
   
2. Update `api/main.py` in `compute_query_embedding()` method:
   - Line ~220: Update API URL
   - Test with your HF token
   
3. Alternative: Use OpenAI embeddings API or local sentence-transformers
   - OpenAI: Fast but costs money
   - Local: Add sentence-transformers to Lambda (but adds ~500MB - too large)

## Testing After Fix
```powershell
# Copy embeddings back to API folder
Copy-Item "data\embeddings\*" "api\data\embeddings\"

# Deploy
sam build; sam deploy

# Test
$token = "your-jwt-token"
$headers = @{Authorization = "Bearer $token"}
$body = @{llava_description="temple shrine";itinerary_landmarks=@("Tokyo Tower");max_distance_km=10;top_k=5} | ConvertTo-Json
Invoke-WebRequest -Uri "https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod/recommend" -Method POST -Headers $headers -Body $body -ContentType "application/json"
```

## Notes
- Current keyword similarity works well for proximity-based recommendations
- Semantic similarity would improve quality by ~10-20% (not critical)
- System is production-ready as-is

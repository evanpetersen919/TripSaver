# Testing the Multi-Tier Landmark Detection

## Test Scenarios

### Test 1: Popular Landmark (Tier 1 Success)
**Image:** Eiffel Tower, Statue of Liberty, Big Ben
**Expected:**
- ✅ EfficientNet detects with >80% confidence
- User accepts → Added to itinerary
- **Tiers 2-3 never called** (efficient!)

### Test 2: Medium Landmark (Tier 2 Success)
**Image:** Regional landmarks, state monuments
**Expected:**
- ⚠️ EfficientNet gives low confidence (<50%)
- User clicks "Analyze Deeper"
- ✅ CLIP + Groq finds match with 65% similarity
- User confirms → Added to itinerary
- **Tier 3 never called** (still free!)

### Test 3: Niche Landmark (Tier 3 Activated)
**Image:** Small local landmarks, obscure places
**Expected:**
- ⚠️ EfficientNet low confidence
- User clicks "Analyze Deeper"
- ⚠️ CLIP finds match but only 45% similarity (< 60%)
- 🚀 **Google Vision API automatically called**
- ✅ Vision API identifies with 85% confidence
- User confirms → Added to itinerary
- **Cost: $0.0015 per image**

### Test 4: Complete Failure (Tier 5)
**Image:** Random objects, abstract art, non-landmarks
**Expected:**
- ❌ All AI methods fail
- Manual entry prompt appears
- User types landmark name
- Fetches photos via Google Places
- Added to itinerary

## Current Status (Before Google Cloud Setup)

**Right now you can test Tiers 1-2:**
```bash
cd d:\VS Code\cv_pipeline\frontend
npm run dev
```

Then:
1. Upload a famous landmark → Should work instantly (Tier 1)
2. Upload a medium landmark → Click "Analyze Deeper" → Should find it (Tier 2)
3. Upload a very niche landmark → Will use manual entry (Tier 5)

**After Google Cloud setup, Tier 3 will automatically activate for niche landmarks.**

## Frontend Testing Checklist

### ✅ Tier 1 - EfficientNet (Already Working)
- [ ] Upload famous landmark image
- [ ] Verify top 5 predictions appear
- [ ] Verify photos load from Google Places
- [ ] Click top prediction
- [ ] Verify added to itinerary with correct location

### ✅ Tier 2 - CLIP + Groq (Already Working)
- [ ] Upload medium landmark
- [ ] Click "Analyze Deeper" button
- [ ] Purple "Thinking Harder" modal appears
- [ ] Confirmation modal shows with AI analysis
- [ ] Click "Yes, Add to Itinerary"
- [ ] Verify location added correctly

### 🔜 Tier 3 - Google Vision API (After Setup)
- [ ] Upload very niche landmark
- [ ] Click "Analyze Deeper"
- [ ] Backend automatically tries Vision API
- [ ] Check Lambda logs for "Google Vision API" message
- [ ] Verify result from Vision API shown in confirmation modal

### ✅ Tier 5 - Manual Entry (Already Working)
- [ ] Cause AI failure (try random image)
- [ ] Verify manual entry prompt appears
- [ ] Enter landmark name
- [ ] Verify photos fetched
- [ ] Verify location added

## Backend Logs to Watch

When testing with Vision API enabled, you'll see:
```
✅ Google Cloud credentials loaded for Vision API
🔍 CLIP similarity: 0.45 (below threshold)
🚀 Triggering Google Vision API (Tier 3)...
✅ Vision API result: Tokyo Station (confidence: 0.89)
```

## Cost Monitoring

Track your Vision API usage:
1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/dashboard)
2. Click "Cloud Vision API"
3. View usage graph

**Expected costs:**
- First 1,000 requests/month: FREE
- Testing (50-100 images): $0 (within free tier)
- Production (1,000 uploads/month, 10% tier 3): ~$0.15/month

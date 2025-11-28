# Instructions for Deploying to Hugging Face Spaces

## Step 1: Create the Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **Owner**: evanpetersen919 (your username)
   - **Space name**: `landmark-detector`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free) - will auto-upgrade to GPU if available
   - **Visibility**: Public

## Step 2: Upload Files

Once the Space is created, upload these files:

1. **app.py** - Main application file
2. **requirements.txt** - Python dependencies
3. **README.md** - Space documentation
4. **landmark_detector_500classes_best.pth** - Trained model (132MB)

### Upload via Git (Recommended for large files)

```bash
# Clone the Space repository
git clone https://huggingface.co/spaces/evanpetersen919/landmark-detector
cd landmark-detector

# Copy files from huggingface_space folder
cp ../huggingface_space/app.py .
cp ../huggingface_space/requirements.txt .
cp ../huggingface_space/README.md .

# Copy the model file
cp ../data/checkpoints/landmark_detector_500classes_best.pth .

# Commit and push
git add .
git commit -m "Initial deployment: EfficientNet-B3 landmark detector"
git push
```

### Upload via Web UI

1. Go to your Space: https://huggingface.co/spaces/evanpetersen919/landmark-detector
2. Click "Files" tab
3. Click "Add file" → "Upload files"
4. Upload all 4 files
5. Commit changes

## Step 3: Wait for Build

- Hugging Face will automatically install dependencies
- Build time: ~3-5 minutes
- The Space will show "Building..." status
- Once complete, status changes to "Running"

## Step 4: Test the Space

1. Visit your Space URL: https://evanpetersen919-landmark-detector.hf.space
2. Upload a test image
3. Verify predictions are returned
4. Check the JSON output format

## Step 5: Get API Endpoint

Your API endpoint will be:
```
https://evanpetersen919-landmark-detector.hf.space/api/predict
```

Use this URL in your AWS Lambda function to call the model.

## Step 6: Test API Call

```python
import requests
from PIL import Image
import io

# Test the API
url = "https://evanpetersen919-landmark-detector.hf.space/api/predict"

# Load test image
image = Image.open("test_landmark.jpg")

# Convert to bytes
buf = io.BytesIO()
image.save(buf, format='JPEG')
buf.seek(0)

# Make request
response = requests.post(url, files={"data": buf})
result = response.json()

print(result)
```

## Troubleshooting

**Build fails:**
- Check requirements.txt for typos
- Ensure torch version is compatible with CUDA

**Model not found:**
- Verify landmark_detector_500classes_best.pth is uploaded
- Check filename matches exactly in app.py

**Out of memory:**
- Space might need GPU upgrade
- Or reduce batch size in inference

**Slow inference:**
- Free tier uses CPU by default
- Can request GPU upgrade (still free for popular Spaces)

## Free Forever?

Yes! Hugging Face Spaces are free as long as:
- Space is public
- Uses reasonable compute (CPU free tier)
- No excessive traffic (fair use)

Popular Spaces may get upgraded to GPU for free automatically.

---

**Next step**: Once deployed, update `api/main.py` in Lambda to call this Space instead of loading the model locally.

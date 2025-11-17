# Computer Vision Pipeline
## Photo Location Detection System

Pure CV focus - parallel model execution for landmark detection, scene classification, and visual similarity search.

## Structure
- `core/` - Pipeline orchestration and configuration
- `models/` - Individual CV models (Scene, CLIP, Landmark)
- `data/` - Sample images and embeddings
- `tests/` - Unit and integration tests
- `demo_cv.py` - Demo script

## Implementation Order
1. models/image_utils.py
2. models/scene_classifier.py
3. models/clip_embedder.py
4. models/landmark_detector.py
5. core/config.py
6. core/vision_pipeline.py
7. demo_cv.py
8. tests/test_vision.py

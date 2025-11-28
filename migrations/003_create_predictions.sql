-- Migration 003: Create Predictions Table
-- Description: Model prediction results and caching
-- Author: Evan Petersen
-- Date: November 2025

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    itinerary_id INTEGER REFERENCES itineraries(id) ON DELETE SET NULL,
    
    -- Image details
    image_url VARCHAR(500),
    image_hash VARCHAR(64),
    
    -- Model predictions
    predicted_landmark VARCHAR(200) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(50),
    
    -- Full results stored as JSON
    top_k_predictions JSONB,
    llava_description TEXT,
    clip_embedding JSONB,
    
    -- Recommendation results
    recommendations JSONB,
    
    -- Performance metrics
    inference_time_ms DOUBLE PRECISION,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_predictions_user_id ON predictions(user_id);
CREATE INDEX idx_predictions_itinerary_id ON predictions(itinerary_id);
CREATE INDEX idx_predictions_image_hash ON predictions(image_hash);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX idx_predictions_confidence ON predictions(confidence);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);

-- Index for JSON searches
CREATE INDEX idx_predictions_top_k ON predictions USING GIN(top_k_predictions);
CREATE INDEX idx_predictions_recommendations ON predictions USING GIN(recommendations);

-- Add check constraint for confidence range
ALTER TABLE predictions
    ADD CONSTRAINT check_confidence_range CHECK (confidence >= 0 AND confidence <= 1);

-- Add comments
COMMENT ON TABLE predictions IS 'Model prediction results with caching for performance';
COMMENT ON COLUMN predictions.user_id IS 'Foreign key to users table';
COMMENT ON COLUMN predictions.itinerary_id IS 'Optional foreign key to itineraries table';
COMMENT ON COLUMN predictions.image_url IS 'S3 URL if image stored in cloud';
COMMENT ON COLUMN predictions.image_hash IS 'SHA-256 hash for deduplication';
COMMENT ON COLUMN predictions.predicted_landmark IS 'Top predicted landmark name';
COMMENT ON COLUMN predictions.confidence IS 'Confidence score (0.0 to 1.0)';
COMMENT ON COLUMN predictions.model_version IS 'Model version identifier (e.g., "500class_v1")';
COMMENT ON COLUMN predictions.top_k_predictions IS 'JSON array of top-5 predictions with confidences';
COMMENT ON COLUMN predictions.llava_description IS 'Natural language description from LLaVA model';
COMMENT ON COLUMN predictions.clip_embedding IS 'JSON array of 512 CLIP embedding values';
COMMENT ON COLUMN predictions.recommendations IS 'JSON array of recommended landmarks';
COMMENT ON COLUMN predictions.inference_time_ms IS 'Total inference time in milliseconds';

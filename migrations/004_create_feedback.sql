-- Migration 004: Create Feedback Table
-- Description: User feedback on model predictions for continuous improvement
-- Author: Evan Petersen
-- Date: November 2025

-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    
    -- Feedback details
    is_correct BOOLEAN NOT NULL,
    corrected_landmark VARCHAR(200),
    confidence_level VARCHAR(20),
    
    -- Additional context
    comments TEXT,
    
    -- Timestamp
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_feedback_user_id ON feedback(user_id);
CREATE INDEX idx_feedback_prediction_id ON feedback(prediction_id);
CREATE INDEX idx_feedback_is_correct ON feedback(is_correct);
CREATE INDEX idx_feedback_confidence_level ON feedback(confidence_level);
CREATE INDEX idx_feedback_created_at ON feedback(created_at);

-- Ensure one feedback per prediction
CREATE UNIQUE INDEX idx_feedback_prediction_unique ON feedback(prediction_id);

-- Add check constraint for confidence level
ALTER TABLE feedback
    ADD CONSTRAINT check_confidence_level CHECK (
        confidence_level IN ('high', 'medium', 'low')
    );

-- Add comments
COMMENT ON TABLE feedback IS 'User feedback on predictions for model improvement';
COMMENT ON COLUMN feedback.user_id IS 'Foreign key to users table';
COMMENT ON COLUMN feedback.prediction_id IS 'Foreign key to predictions table';
COMMENT ON COLUMN feedback.is_correct IS 'Whether the prediction was correct';
COMMENT ON COLUMN feedback.corrected_landmark IS 'Correct landmark name if prediction was wrong';
COMMENT ON COLUMN feedback.confidence_level IS 'Confidence level: high, medium, or low';
COMMENT ON COLUMN feedback.comments IS 'Optional user comments';

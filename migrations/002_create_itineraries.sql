-- Migration 002: Create Itineraries Table
-- Description: User travel plans and landmark collections
-- Author: Evan Petersen
-- Date: November 2025

-- Create itineraries table
CREATE TABLE IF NOT EXISTS itineraries (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Itinerary details
    name VARCHAR(200) NOT NULL,
    landmarks JSONB NOT NULL,
    description TEXT,
    
    -- Geographic center point for proximity search
    center_lat DOUBLE PRECISION,
    center_lon DOUBLE PRECISION,
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_itineraries_user_id ON itineraries(user_id);
CREATE INDEX idx_itineraries_is_active ON itineraries(is_active);
CREATE INDEX idx_itineraries_created_at ON itineraries(created_at);
CREATE INDEX idx_itineraries_center_location ON itineraries(center_lat, center_lon);

-- Index for JSONB landmark searches
CREATE INDEX idx_itineraries_landmarks ON itineraries USING GIN(landmarks);

-- Auto-update trigger
CREATE TRIGGER update_itineraries_updated_at
    BEFORE UPDATE ON itineraries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments
COMMENT ON TABLE itineraries IS 'User travel itineraries with landmark collections';
COMMENT ON COLUMN itineraries.user_id IS 'Foreign key to users table';
COMMENT ON COLUMN itineraries.name IS 'Itinerary name (e.g., "Paris Trip 2025")';
COMMENT ON COLUMN itineraries.landmarks IS 'JSON array of landmark names';
COMMENT ON COLUMN itineraries.center_lat IS 'Center latitude for proximity calculations';
COMMENT ON COLUMN itineraries.center_lon IS 'Center longitude for proximity calculations';
COMMENT ON COLUMN itineraries.is_active IS 'Active/archived status';

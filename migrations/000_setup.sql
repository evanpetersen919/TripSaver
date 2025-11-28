-- Migration 000: Initial Database Setup
-- Description: Create database and enable extensions
-- Author: Evan Petersen
-- Date: November 2025

-- Create database (run this as postgres superuser)
-- CREATE DATABASE cv_location_db;

-- Connect to the database
-- \c cv_location_db

-- Enable UUID extension (optional, for future use)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for password hashing (optional)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enable pg_trgm for fuzzy text search (optional)
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Check PostgreSQL version
SELECT version();

-- Verify extensions
SELECT * FROM pg_extension;

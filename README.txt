SMART HERD - RENDER FLAT DEPLOYMENT

This package intentionally contains NO app/ or models/ directories.
It is designed for Render's direct "Add files via upload" workflow.

Upload every file in this folder together.

Required Render environment variables:
  SUPABASE_URL
  SUPABASE_SECRET_KEY
  INTERNAL_API_KEY

Recommended:
  FARM_TIMEZONE=Africa/Nairobi
  MODEL2_VERSION=model2-public-xgb-v1-shadow

Health check:
  /health

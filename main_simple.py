"""
ChefoodAI AI Service - Simplified Main API Server
Minimal configuration for Cloud Run deployment debugging
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import time

# Create FastAPI app
app = FastAPI(
    title="ChefoodAI AI Service",
    version="1.0.0",
    description="AI Service API for ChefoodAI"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ChefoodAI AI Service",
        "status": "running",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ChefoodAI AI Service",
        "port": os.environ.get("PORT", 8000),
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "timestamp": time.time()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {
        "status": "ready",
        "service": "ChefoodAI AI Service",
        "timestamp": time.time()
    }

@app.post("/api/ai/meal-plan/generate")
async def generate_meal_plan_stub(request: Request):
    """Stub endpoint for meal plan generation"""
    return {
        "status": "success",
        "message": "This is a stub response - AI service simplified for deployment",
        "meal_plan": {
            "id": "stub-plan-123",
            "meals": []
        }
    }

@app.post("/api/ai/recipe/enhance")
async def enhance_recipe_stub(request: Request):
    """Stub endpoint for recipe enhancement"""
    return {
        "status": "success",
        "message": "This is a stub response - AI service simplified for deployment",
        "enhanced_recipe": {}
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
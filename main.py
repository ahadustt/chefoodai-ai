"""
ChefoodAI AI Service - Main Entry Point
Production-ready AI service with comprehensive features
"""

# Import the application from the new API structure
from api import create_app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    from config import get_settings
    
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.value.lower(),
        access_log=settings.enable_access_logs,
        workers=1,  # Single worker for AI services to avoid memory issues
        loop="asyncio",
        timeout_keep_alive=settings.keep_alive_timeout,
        limit_concurrency=settings.max_concurrent_requests,
    )
#!/usr/bin/env python3
"""Startup script for ChefoodAI AI Service - Ensures proper Cloud Run deployment"""

import os
import sys
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_server():
    """Start the FastAPI server with proper configuration for Cloud Run"""
    
    # Get port from environment variable (Cloud Run requirement)
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Must bind to all interfaces
    
    logger.info(f"Starting ChefoodAI AI Service")
    logger.info(f"Port: {port}")
    logger.info(f"Host: {host}")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'production')}")
    logger.info(f"Google Cloud Project: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'not set')}")
    
    try:
        # Import the app here to catch any import errors
        from main import app
        logger.info("Successfully imported FastAPI app")
        
        # Configure uvicorn with minimal settings for reliability
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            use_colors=False,  # Disable colors for Cloud Run logs
            server_header=False,  # Don't expose server info
            date_header=False,  # Cloud Run adds its own headers
            limit_concurrency=500,  # Lower limit for AI service
            timeout_keep_alive=5,  # Short keep-alive for Cloud Run
            loop="auto"  # Let uvicorn choose the best loop
        )
        
        server = uvicorn.Server(config)
        logger.info(f"Server configured, starting on {host}:{port}")
        server.run()
        
    except ImportError as e:
        logger.error(f"Failed to import app: {e}")
        logger.error("Make sure main.py exists and has 'app' variable")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    start_server()
"""
Health check and monitoring routes
Comprehensive system health monitoring and diagnostics
"""

import time
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import structlog

from config import get_settings
from ..models import HealthResponse
from ..middleware import get_metrics

logger = structlog.get_logger()
router = APIRouter(tags=["Health & Monitoring"])

# Track service start time for uptime calculation
SERVICE_START_TIME = time.time()


async def check_vertex_ai_health() -> bool:
    """Check Vertex AI service health"""
    try:
        # Import here to avoid circular imports
        from ai.vertex_integration import VertexAIService
        
        # Quick health check - this could be a simple model call
        # For now, just check if we can import and initialize
        service = VertexAIService(
            project_id=get_settings().google_cloud_project,
            location=get_settings().vertex_ai_location
        )
        return True
    except Exception as e:
        logger.warning("Vertex AI health check failed", error=str(e))
        return False


async def check_redis_health() -> bool:
    """Check Redis connection health"""
    try:
        if not get_settings().redis_url:
            return True  # Redis is optional
        
        # Import here to avoid issues if redis is not available
        import redis.asyncio as redis_async
        
        redis_client = redis_async.from_url(
            get_settings().redis_url,
            socket_timeout=2,
            socket_connect_timeout=2
        )
        
        # Simple ping test
        await redis_client.ping()
        await redis_client.close()
        return True
    except Exception as e:
        logger.warning("Redis health check failed", error=str(e))
        return False


async def check_storage_health() -> bool:
    """Check Google Cloud Storage health"""
    try:
        from google.cloud import storage
        
        client = storage.Client(project=get_settings().google_cloud_project)
        # Simple list operation to check connectivity
        list(client.list_buckets(max_results=1))
        return True
    except Exception as e:
        logger.warning("Storage health check failed", error=str(e))
        return False


async def check_openai_health() -> bool:
    """Check OpenAI API health"""
    try:
        if not get_settings().openai_api_key:
            return True  # OpenAI is optional
        
        import openai
        openai.api_key = get_settings().openai_api_key
        
        # Quick models list to check API connectivity
        # This is a lightweight call
        client = openai.OpenAI(api_key=get_settings().openai_api_key)
        models = client.models.list()
        return True
    except Exception as e:
        logger.warning("OpenAI health check failed", error=str(e))
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint
    Checks all critical dependencies and returns detailed status
    """
    start_time = time.time()
    settings = get_settings()
    
    # Run all health checks concurrently for faster response
    checks_tasks = {
        "vertex_ai": check_vertex_ai_health(),
        "redis": check_redis_health(),
        "storage": check_storage_health(),
        "openai": check_openai_health(),
    }
    
    # Execute all checks concurrently
    check_results = {}
    for name, task in checks_tasks.items():
        try:
            check_results[name] = await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for {name}")
            check_results[name] = False
        except Exception as e:
            logger.error(f"Health check error for {name}", error=str(e))
            check_results[name] = False
    
    # Calculate uptime
    uptime = time.time() - SERVICE_START_TIME
    
    # Determine overall health status
    all_healthy = all(check_results.values())
    status = "healthy" if all_healthy else "unhealthy"
    
    # Create response
    response = HealthResponse(
        status=status,
        service="chefoodai-ai-service",
        version=settings.app_version,
        checks=check_results,
        uptime=round(uptime, 2)
    )
    
    # Return appropriate HTTP status code
    status_code = 200 if all_healthy else 503
    
    logger.info(
        "Health check completed",
        status=status,
        checks=check_results,
        uptime=uptime,
        check_duration=round(time.time() - start_time, 3)
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response.dict()
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint
    Simple check to determine if service is ready to accept traffic
    """
    settings = get_settings()
    
    # Basic readiness checks
    ready = True
    checks = {}
    
    # Check if we can create AI service instances
    try:
        from ai.vertex_integration import VertexAIService
        checks["vertex_ai_init"] = True
    except Exception:
        checks["vertex_ai_init"] = False
        ready = False
    
    # Check configuration
    checks["config_valid"] = bool(settings.google_cloud_project)
    if not checks["config_valid"]:
        ready = False
    
    status_code = 200 if ready else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if ready else "not_ready",
            "service": "chefoodai-ai-service",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@router.get("/metrics")
async def get_application_metrics():
    """
    Application metrics endpoint
    Returns detailed metrics about service performance and usage
    """
    settings = get_settings()
    
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    # Get metrics from middleware
    metrics = get_metrics()
    
    # Add additional system metrics
    uptime = time.time() - SERVICE_START_TIME
    
    enhanced_metrics = {
        **metrics,
        "service_info": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "uptime_seconds": round(uptime, 2),
            "start_time": datetime.fromtimestamp(SERVICE_START_TIME).isoformat()
        },
        "configuration": {
            "rate_limit_requests": settings.rate_limit_requests,
            "rate_limit_window": settings.rate_limit_window,
            "max_request_size": settings.max_request_size,
            "request_timeout": settings.request_timeout,
            "features": {
                "cost_optimization": settings.enable_cost_optimization,
                "fallback_strategies": settings.enable_fallback_strategies,
                "ab_testing": settings.enable_ab_testing,
                "image_generation": settings.enable_image_generation,
                "advanced_nutrition": settings.enable_advanced_nutrition
            }
        }
    }
    
    return enhanced_metrics


@router.get("/")
async def root():
    """
    Root endpoint with service information
    """
    settings = get_settings()
    uptime = time.time() - SERVICE_START_TIME
    
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
        "uptime_seconds": round(uptime, 2),
        "capabilities": [
            "recipe_generation",
            "image_generation", 
            "nutrition_analysis",
            "meal_plan_optimization",
            "shopping_list_optimization",
            "dietary_validation",
            "ingredient_substitution"
        ],
        "api_documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics" if settings.enable_metrics else None,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ping")
async def ping():
    """
    Simple ping endpoint for basic connectivity testing
    """
    return {
        "message": "pong",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "chefoodai-ai-service"
    }
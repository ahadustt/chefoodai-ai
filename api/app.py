"""
Main FastAPI application factory
Comprehensive production-ready application setup
"""

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
import structlog

from config import get_settings, validate_production_config
from .models import ErrorResponse, ErrorType, ResponseStatus
from .middleware import (
    RequestTracingMiddleware,
    RateLimitMiddleware, 
    SecurityHeadersMiddleware,
    RequestSizeMiddleware,
    MetricsMiddleware,
    TimeoutMiddleware,
    metrics_middleware_instance
)
from .routes import (
    health_router,
    recipes_router,
    # Import other routers as they're created
)

logger = structlog.get_logger()

# Global variables for AI services
vertex_ai = None
dalle_service = None
gemini_service = None
nutrition_analyzer = None
meal_plan_optimizer = None
recipe_generator = None
shopping_optimizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    settings = get_settings()
    
    logger.info("Starting ChefoodAI AI Service", version=settings.app_version)
    
    # Validate production configuration
    if settings.is_production:
        config_issues = validate_production_config(settings)
        if config_issues:
            logger.error("Production configuration issues", issues=config_issues)
            for issue in config_issues:
                logger.error(f"Configuration issue: {issue}")
    
    # Initialize AI services
    global vertex_ai, dalle_service, gemini_service
    global nutrition_analyzer, meal_plan_optimizer, recipe_generator, shopping_optimizer
    
    try:
        # Initialize Vertex AI service
        from ai.vertex_integration import VertexAIService
        vertex_ai = VertexAIService(
            project_id=settings.google_cloud_project,
            location=settings.vertex_ai_location
        )
        logger.info("Vertex AI service initialized")
        
        # Initialize optional AI services
        if settings.openai_api_key and settings.enable_image_generation:
            try:
                from ai.dalle_service import DalleService
                dalle_service = DalleService(api_key=settings.openai_api_key)
                logger.info("DALL-E service initialized")
            except ImportError:
                logger.warning("DALL-E service not available")
        
        # Initialize Gemini service
        try:
            from ai.gemini_service import GeminiService
            gemini_service = GeminiService(vertex_ai)
            logger.info("Gemini service initialized")
        except ImportError:
            logger.warning("Gemini service not available")
        
        # Initialize specialized AI services
        if settings.enable_advanced_nutrition:
            try:
                from ai.nutrition_analyzer import NutritionAnalyzer
                nutrition_analyzer = NutritionAnalyzer(vertex_ai)
                logger.info("Nutrition analyzer initialized")
            except ImportError:
                logger.warning("Nutrition analyzer not available")
        
        try:
            from ai.meal_plan_optimizer import MealPlanOptimizer
            meal_plan_optimizer = MealPlanOptimizer(vertex_ai)
            logger.info("Meal plan optimizer initialized")
        except ImportError:
            logger.warning("Meal plan optimizer not available")
        
        try:
            from ai.recipe_generator import RecipeGenerator
            recipe_generator = RecipeGenerator(vertex_ai)
            logger.info("Recipe generator initialized")
        except ImportError:
            logger.warning("Recipe generator not available")
        
        try:
            from ai.shopping_optimizer import ShoppingOptimizer
            shopping_optimizer = ShoppingOptimizer(vertex_ai)
            logger.info("Shopping optimizer initialized")
        except ImportError:
            logger.warning("Shopping optimizer not available")
        
        logger.info("AI services initialization complete")
        
    except Exception as e:
        logger.error("Failed to initialize AI services", error=str(e))
        # Don't fail startup - services can run in fallback mode
    
    yield
    
    # Shutdown
    logger.info("Shutting down ChefoodAI AI Service")
    
    # Cleanup AI services if needed
    try:
        if vertex_ai:
            await vertex_ai.cleanup()
    except Exception as e:
        logger.warning("Error during AI service cleanup", error=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="Advanced AI-powered features for ChefoodAI",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )
    
    # Add middleware (order matters!)
    
    # 1. CORS Middleware (should be first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # 2. Security Headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 3. Request Size Limiting
    app.add_middleware(RequestSizeMiddleware)
    
    # 4. Rate Limiting
    app.add_middleware(RateLimitMiddleware)
    
    # 5. Request Timeout
    app.add_middleware(TimeoutMiddleware)
    
    # 6. Metrics Collection
    global metrics_middleware_instance
    metrics_middleware_instance = MetricsMiddleware(app)
    app.add_middleware(MetricsMiddleware)
    
    # 7. Request Tracing (should be last)
    app.add_middleware(RequestTracingMiddleware)
    
    # Exception handlers
    
    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        """Custom HTTP exception handler with standardized error response"""
        request_id = getattr(request.state, 'request_id', None)
        
        # Determine error type based on status code
        error_type_map = {
            400: ErrorType.VALIDATION_ERROR,
            401: ErrorType.AUTHENTICATION_ERROR,
            403: ErrorType.AUTHENTICATION_ERROR,
            404: ErrorType.VALIDATION_ERROR,
            422: ErrorType.VALIDATION_ERROR,
            429: ErrorType.RATE_LIMIT_EXCEEDED,
            500: ErrorType.INTERNAL_ERROR,
            503: ErrorType.SERVICE_UNAVAILABLE,
            504: ErrorType.TIMEOUT_ERROR,
        }
        
        error_response = ErrorResponse(
            error_type=error_type_map.get(exc.status_code, ErrorType.INTERNAL_ERROR),
            detail=str(exc.detail),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id} if request_id else {}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Custom validation exception handler"""
        request_id = getattr(request.state, 'request_id', None)
        
        # Format validation errors
        errors = []
        for error in exc.errors():
            errors.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        error_response = ErrorResponse(
            error_type=ErrorType.VALIDATION_ERROR,
            detail="Request validation failed",
            errors=errors,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id} if request_id else {}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Catch-all exception handler"""
        request_id = getattr(request.state, 'request_id', None)
        
        logger.error(
            "Unhandled exception",
            error=str(exc),
            error_type=type(exc).__name__,
            request_id=request_id,
            path=request.url.path if hasattr(request, 'url') else None
        )
        
        # Don't expose internal error details in production
        detail = str(exc) if not settings.is_production else "Internal server error"
        
        error_response = ErrorResponse(
            error_type=ErrorType.INTERNAL_ERROR,
            detail=detail,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id} if request_id else {}
        )
    
    # Register routers
    app.include_router(health_router)
    app.include_router(recipes_router)
    
    # Add other routers as they're created:
    # app.include_router(images_router)
    # app.include_router(nutrition_router)
    # app.include_router(meal_plans_router)
    # app.include_router(shopping_router)
    
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            level=getattr(structlog.stdlib, settings.log_level.value)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add startup event for additional initialization
    @app.on_event("startup")
    async def startup_event():
        """Additional startup tasks"""
        logger.info("Application startup complete")
        
        # Warm up AI services if needed
        if settings.is_production:
            asyncio.create_task(warmup_ai_services())
    
    return app


async def warmup_ai_services():
    """Warm up AI services to reduce cold start latency"""
    try:
        logger.info("Warming up AI services")
        
        # Make a simple test call to each service
        if vertex_ai:
            # Test Vertex AI connection
            pass
        
        if recipe_generator:
            # Could make a simple test recipe generation
            pass
        
        logger.info("AI services warmup complete")
        
    except Exception as e:
        logger.warning("AI services warmup failed", error=str(e))


# Create the app instance
app = create_app()
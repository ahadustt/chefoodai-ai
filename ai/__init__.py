"""
ChefoodAI - Google Vertex AI Integration Package
Complete AI-powered recipe generation and meal planning system
"""

import os
import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Only import what's essential - make other imports conditional
from .vertex_integration import VertexAIService, RecipeRequest, ModelVersion

logger = logging.getLogger(__name__)

# Optional imports - only import if needed
def get_redis_async():
    """Conditionally import redis async client"""
    try:
        from redis import asyncio as redis_async
        return redis_async
    except ImportError:
        logger.warning("redis async not available - caching disabled")
        return None

def get_cost_optimizer():
    """Conditionally import cost optimization"""
    try:
        from .cost_optimization import CostOptimizer, CostTier
        return CostOptimizer, CostTier
    except ImportError:
        logger.warning("Cost optimization not available")
        return None, None

def get_fallback_manager():
    """Conditionally import fallback strategies"""
    try:
        from .fallback_strategies import FallbackManager, FallbackConfig
        return FallbackManager, FallbackConfig
    except ImportError:
        logger.warning("Fallback strategies not available")
        return None, None

def get_ab_testing():
    """Conditionally import A/B testing"""
    try:
        from .ab_testing import ABTestManager, Experiment, ExperimentVariant, ExperimentStatus, MetricType
        return ABTestManager, Experiment, ExperimentVariant, ExperimentStatus, MetricType
    except ImportError:
        logger.warning("A/B testing not available")
        return None, None, None, None, None

class ChefoodAIConfig:
    """Central configuration for ChefoodAI system"""
    
    def __init__(self):
        # Vertex AI Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "mychef-467404")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        
        # Redis Configuration (optional)
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        
        # Cost Optimization
        self.cost_optimization_enabled = os.getenv("COST_OPTIMIZATION_ENABLED", "true").lower() == "true"
        self.max_daily_spend = float(os.getenv("MAX_DAILY_SPEND", "100.0"))
        
        # Fallback Configuration
        self.fallback_enabled = os.getenv("FALLBACK_ENABLED", "true").lower() == "true"
        self.fallback_timeout = int(os.getenv("FALLBACK_TIMEOUT", "30"))
        
        # A/B Testing
        self.ab_testing_enabled = os.getenv("AB_TESTING_ENABLED", "false").lower() == "true"
        
        # Caching
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
        
        # Rate Limiting
        self.rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_to_cloud = os.getenv("LOG_TO_CLOUD", "true").lower() == "true"

# Global configuration instance
config = ChefoodAIConfig()

class ChefoodAI:
    """
    Main ChefoodAI service class - simplified for deployment
    Provides a unified interface to all AI capabilities
    """
    
    def __init__(self, config: Optional[ChefoodAIConfig] = None):
        self.config = config or ChefoodAIConfig()
        self.vertex_service = None
        self._redis_client = None
        self._initialized = False
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"ChefoodAI initialized for project: {self.config.project_id}")
    
    async def initialize(self):
        """Initialize all services"""
        if self._initialized:
            return
            
        try:
            # Initialize Vertex AI service
            self.vertex_service = VertexAIService(
                project_id=self.config.project_id,
                location=self.config.location
            )
            
            # Initialize Redis if available and enabled
            if self.config.cache_enabled:
                aioredis = get_aioredis()
                if aioredis:
                    try:
                        self._redis_client = await aioredis.from_url(
                            f"redis://{self.config.redis_host}:{self.config.redis_port}",
                            password=self.config.redis_password,
                            decode_responses=True
                        )
                        logger.info("Redis cache initialized")
                    except Exception as e:
                        logger.warning(f"Redis connection failed: {e}")
                        self._redis_client = None
            
            self._initialized = True
            logger.info("ChefoodAI fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChefoodAI: {e}")
            # Continue without optional services
            self._initialized = True
    
    async def generate_recipe(
        self,
        request: RecipeRequest,
        model_version: ModelVersion = ModelVersion.GEMINI_1_5_PRO,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Generate a recipe using Vertex AI"""
        if not self._initialized:
            await self.initialize()
        
        if not self.vertex_service:
            logger.error("Vertex AI service not available")
            return None
        
        try:
            # Try to get from cache first
            cache_key = None
            if use_cache and self._redis_client:
                cache_key = f"recipe:{hash(str(request))}"
                try:
                    cached_result = await self._redis_client.get(cache_key)
                    if cached_result:
                        logger.info("Recipe retrieved from cache")
                        return json.loads(cached_result)
                except Exception as e:
                    logger.warning(f"Cache read failed: {e}")
            
            # Generate recipe
            recipe = await self.vertex_service.generate_recipe(
                request=request,
                model_version=model_version,
                use_cache=use_cache
            )
            
            # Cache the result
            if recipe and cache_key and self._redis_client:
                try:
                    await self._redis_client.setex(
                        cache_key,
                        self.config.cache_ttl,
                        json.dumps(recipe)
                    )
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")
            
            return recipe
            
        except Exception as e:
            logger.error(f"Recipe generation failed: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        if not self._initialized:
            await self.initialize()
        
        health_info = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "vertex_ai": bool(self.vertex_service),
                "redis_cache": bool(self._redis_client),
            },
            "config": {
                "project_id": self.config.project_id,
                "location": self.config.location,
                "cache_enabled": self.config.cache_enabled,
                "cost_optimization": self.config.cost_optimization_enabled,
            }
        }
        
        return health_info
    
    async def close(self):
        """Clean up resources"""
        if self._redis_client:
            await self._redis_client.close()
        logger.info("ChefoodAI services closed")

# Global instance
chefood_ai = ChefoodAI(config)

# Dependency for FastAPI
async def get_chefood_ai() -> ChefoodAI:
    """FastAPI dependency to get ChefoodAI instance"""
    if not chefood_ai._initialized:
        await chefood_ai.initialize()
    return chefood_ai

# Export key components
__all__ = [
    'ChefoodAI',
    'ChefoodAIConfig', 
    'VertexAIService',
    'RecipeRequest',
    'ModelVersion',
    'chefood_ai',
    'get_chefood_ai',
    'config'
]
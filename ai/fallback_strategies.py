"""
Fallback strategies and resilience patterns for Vertex AI integration
Handles API failures, quota limits, and service degradation gracefully
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import aiohttp
import random

from .vertex_integration import RecipeRequest, MealPlanRequest, ModelVersion

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of failures we need to handle"""
    API_TIMEOUT = "api_timeout"
    QUOTA_EXCEEDED = "quota_exceeded"
    SAFETY_BLOCK = "safety_block"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION_ERROR = "auth_error"

class FallbackLevel(Enum):
    """Levels of fallback degradation"""
    RETRY_SAME = "retry_same"           # Retry with same model
    DOWNGRADE_MODEL = "downgrade_model"  # Use cheaper/faster model
    USE_CACHE = "use_cache"             # Force cache lookup
    TEMPLATE_RESPONSE = "template"       # Use template-based response
    STATIC_CONTENT = "static"           # Return pre-defined content
    EXTERNAL_API = "external"           # Use alternative API

@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    max_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 30.0
    enable_model_downgrade: bool = True
    enable_cache_fallback: bool = True
    enable_template_fallback: bool = True
    enable_external_fallback: bool = False
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

@dataclass
class FailureContext:
    """Context information about a failure"""
    failure_type: FailureType
    error_message: str
    timestamp: datetime
    model_used: str
    request_data: Dict[str, Any]
    attempt_number: int
    user_id: Optional[str] = None

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout_duration: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.timeout_duration):
                self.state = "half-open"
                return True
            return False
        
        # half-open state
        return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class FallbackManager:
    """Main fallback and resilience manager"""
    
    def __init__(self, config: FallbackConfig = None):
        self.config = config or FallbackConfig()
        self.circuit_breakers = {}  # Per-model circuit breakers
        self.failure_history = []
        self.static_recipes = self._load_static_recipes()
        self.template_engine = RecipeTemplateEngine()
        
    def get_circuit_breaker(self, model_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for model"""
        if model_name not in self.circuit_breakers:
            self.circuit_breakers[model_name] = CircuitBreaker(
                self.config.circuit_breaker_threshold,
                self.config.circuit_breaker_timeout
            )
        return self.circuit_breakers[model_name]
    
    async def execute_with_fallback(
        self,
        primary_func: Callable,
        fallback_context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute function with comprehensive fallback strategy"""
        
        model_name = kwargs.get("model_version", "unknown")
        circuit_breaker = self.get_circuit_breaker(model_name)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {model_name}, using fallback")
            return await self._execute_fallback(
                FallbackLevel.DOWNGRADE_MODEL,
                fallback_context,
                FailureContext(
                    FailureType.SERVICE_UNAVAILABLE,
                    "Circuit breaker open",
                    datetime.utcnow(),
                    model_name,
                    kwargs,
                    0
                )
            )
        
        # Try primary function with retries
        for attempt in range(self.config.max_retries):
            try:
                result = await primary_func(*args, **kwargs)
                
                # Success - reset circuit breaker
                circuit_breaker.record_success()
                
                # Track successful execution
                await self._track_success(model_name, attempt + 1)
                
                return result
                
            except Exception as e:
                failure_type = self._classify_error(e)
                failure_context = FailureContext(
                    failure_type=failure_type,
                    error_message=str(e),
                    timestamp=datetime.utcnow(),
                    model_used=model_name,
                    request_data=kwargs,
                    attempt_number=attempt + 1,
                    user_id=fallback_context.get("user_id")
                )
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {failure_type.value} - {str(e)}"
                )
                
                # Record failure
                circuit_breaker.record_failure()
                self.failure_history.append(failure_context)
                
                # Determine if we should retry or fallback
                if not self._should_retry(failure_type, attempt + 1):
                    return await self._execute_fallback_sequence(
                        failure_context,
                        fallback_context
                    )
                
                # Wait before retry
                if attempt < self.config.max_retries - 1:
                    delay = min(
                        self.config.retry_delay_base * (2 ** attempt) + random.uniform(0, 1),
                        self.config.retry_delay_max
                    )
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        final_failure = FailureContext(
            FailureType.SERVICE_UNAVAILABLE,
            "All retry attempts exhausted",
            datetime.utcnow(),
            model_name,
            kwargs,
            self.config.max_retries
        )
        
        return await self._execute_fallback_sequence(final_failure, fallback_context)
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for appropriate fallback strategy"""
        
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureType.API_TIMEOUT
        elif "quota" in error_str or "rate limit" in error_str:
            return FailureType.QUOTA_EXCEEDED
        elif "safety" in error_str or "blocked" in error_str:
            return FailureType.SAFETY_BLOCK
        elif "network" in error_str or "connection" in error_str:
            return FailureType.NETWORK_ERROR
        elif "auth" in error_str or "permission" in error_str:
            return FailureType.AUTHENTICATION_ERROR
        elif "503" in error_str or "unavailable" in error_str:
            return FailureType.SERVICE_UNAVAILABLE
        else:
            return FailureType.INVALID_RESPONSE
    
    def _should_retry(self, failure_type: FailureType, attempt: int) -> bool:
        """Determine if we should retry based on failure type and attempt"""
        
        # Don't retry authentication errors
        if failure_type == FailureType.AUTHENTICATION_ERROR:
            return False
        
        # Don't retry safety blocks (they'll likely persist)
        if failure_type == FailureType.SAFETY_BLOCK:
            return False
        
        # Retry transient errors
        if failure_type in [
            FailureType.API_TIMEOUT,
            FailureType.NETWORK_ERROR,
            FailureType.SERVICE_UNAVAILABLE
        ]:
            return attempt < self.config.max_retries
        
        # Retry quota errors with exponential backoff
        if failure_type == FailureType.QUOTA_EXCEEDED:
            return attempt < 2  # Only retry once for quota
        
        return attempt < self.config.max_retries
    
    async def _execute_fallback_sequence(
        self,
        failure_context: FailureContext,
        fallback_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fallback sequence based on failure type"""
        
        # Define fallback sequence based on failure type
        fallback_sequence = self._get_fallback_sequence(failure_context.failure_type)
        
        for fallback_level in fallback_sequence:
            try:
                result = await self._execute_fallback(
                    fallback_level,
                    fallback_context,
                    failure_context
                )
                
                if result:
                    # Mark result as fallback
                    result["_fallback_used"] = True
                    result["_fallback_level"] = fallback_level.value
                    result["_original_error"] = failure_context.error_message
                    
                    await self._track_fallback_success(fallback_level)
                    return result
                    
            except Exception as e:
                logger.error(f"Fallback {fallback_level.value} failed: {str(e)}")
                continue
        
        # All fallbacks failed - return error response
        return {
            "error": "All fallback strategies exhausted",
            "original_error": failure_context.error_message,
            "failure_type": failure_context.failure_type.value,
            "timestamp": failure_context.timestamp.isoformat()
        }
    
    def _get_fallback_sequence(self, failure_type: FailureType) -> List[FallbackLevel]:
        """Get appropriate fallback sequence for failure type"""
        
        base_sequence = []
        
        if failure_type == FailureType.QUOTA_EXCEEDED:
            # For quota issues, try cache first, then downgrade
            if self.config.enable_cache_fallback:
                base_sequence.append(FallbackLevel.USE_CACHE)
            if self.config.enable_model_downgrade:
                base_sequence.append(FallbackLevel.DOWNGRADE_MODEL)
        
        elif failure_type == FailureType.SAFETY_BLOCK:
            # For safety blocks, try template or static content
            if self.config.enable_template_fallback:
                base_sequence.append(FallbackLevel.TEMPLATE_RESPONSE)
            base_sequence.append(FallbackLevel.STATIC_CONTENT)
        
        elif failure_type in [FailureType.API_TIMEOUT, FailureType.SERVICE_UNAVAILABLE]:
            # For service issues, try cache, then alternative approaches
            if self.config.enable_cache_fallback:
                base_sequence.append(FallbackLevel.USE_CACHE)
            if self.config.enable_external_fallback:
                base_sequence.append(FallbackLevel.EXTERNAL_API)
            if self.config.enable_template_fallback:
                base_sequence.append(FallbackLevel.TEMPLATE_RESPONSE)
        
        else:
            # Default sequence
            if self.config.enable_cache_fallback:
                base_sequence.append(FallbackLevel.USE_CACHE)
            if self.config.enable_model_downgrade:
                base_sequence.append(FallbackLevel.DOWNGRADE_MODEL)
            if self.config.enable_template_fallback:
                base_sequence.append(FallbackLevel.TEMPLATE_RESPONSE)
        
        # Always add static content as last resort
        base_sequence.append(FallbackLevel.STATIC_CONTENT)
        
        return base_sequence
    
    async def _execute_fallback(
        self,
        fallback_level: FallbackLevel,
        context: Dict[str, Any],
        failure_context: FailureContext
    ) -> Optional[Dict[str, Any]]:
        """Execute specific fallback strategy"""
        
        if fallback_level == FallbackLevel.USE_CACHE:
            return await self._cache_fallback(context, failure_context)
        
        elif fallback_level == FallbackLevel.DOWNGRADE_MODEL:
            return await self._model_downgrade_fallback(context, failure_context)
        
        elif fallback_level == FallbackLevel.TEMPLATE_RESPONSE:
            return await self._template_fallback(context, failure_context)
        
        elif fallback_level == FallbackLevel.STATIC_CONTENT:
            return await self._static_fallback(context, failure_context)
        
        elif fallback_level == FallbackLevel.EXTERNAL_API:
            return await self._external_api_fallback(context, failure_context)
        
        return None
    
    async def _cache_fallback(
        self,
        context: Dict[str, Any],
        failure_context: FailureContext
    ) -> Optional[Dict[str, Any]]:
        """Try to find similar content in cache"""
        
        # This would integrate with the cost optimizer's intelligent cache lookup
        logger.info("Attempting cache fallback")
        
        # Simplified cache lookup - in practice, use intelligent matching
        cache_manager = context.get("cache_manager")
        if cache_manager:
            return await cache_manager.intelligent_cache_lookup(
                failure_context.request_data,
                context.get("request_type", "recipe")
            )
        
        return None
    
    async def _model_downgrade_fallback(
        self,
        context: Dict[str, Any],
        failure_context: FailureContext
    ) -> Optional[Dict[str, Any]]:
        """Retry with a cheaper/faster model"""
        
        current_model = failure_context.model_used
        
        # Model downgrade hierarchy
        downgrade_map = {
            "gemini-1.5-pro": "gemini-1.5-flash",
            "gemini-1.5-flash": "gemini-1.0-pro",
        }
        
        fallback_model = downgrade_map.get(current_model)
        if not fallback_model:
            return None
        
        logger.info(f"Downgrading from {current_model} to {fallback_model}")
        
        # Retry with fallback model
        ai_service = context.get("ai_service")
        if ai_service and context.get("request_type") == "recipe":
            try:
                request = RecipeRequest(**failure_context.request_data)
                return await ai_service.generate_recipe(
                    request,
                    ModelVersion(fallback_model),
                    use_cache=True
                )
            except Exception as e:
                logger.error(f"Model downgrade failed: {str(e)}")
        
        return None
    
    async def _template_fallback(
        self,
        context: Dict[str, Any],
        failure_context: FailureContext
    ) -> Optional[Dict[str, Any]]:
        """Generate response using templates"""
        
        request_data = failure_context.request_data
        request_type = context.get("request_type", "recipe")
        
        if request_type == "recipe":
            return await self.template_engine.generate_recipe_from_template(
                ingredients=request_data.get("ingredients", []),
                dietary_preferences=request_data.get("dietary_preferences", []),
                cuisine_type=request_data.get("cuisine_type"),
                cooking_time=request_data.get("cooking_time_minutes")
            )
        
        elif request_type == "meal_plan":
            return await self.template_engine.generate_meal_plan_from_template(
                days=request_data.get("days", 7),
                dietary_preferences=request_data.get("dietary_preferences", [])
            )
        
        return None
    
    async def _static_fallback(
        self,
        context: Dict[str, Any],
        failure_context: FailureContext
    ) -> Optional[Dict[str, Any]]:
        """Return pre-defined static content"""
        
        request_data = failure_context.request_data
        ingredients = request_data.get("ingredients", [])
        
        # Find best matching static recipe
        if ingredients:
            return self._find_best_static_recipe(ingredients)
        
        # Return default recipe
        return self.static_recipes.get("default")
    
    async def _external_api_fallback(
        self,
        context: Dict[str, Any],
        failure_context: FailureContext
    ) -> Optional[Dict[str, Any]]:
        """Use external API as fallback (e.g., OpenAI, Claude)"""
        
        # This would integrate with alternative AI services
        logger.info("Attempting external API fallback")
        
        # Example integration with alternative service
        try:
            # Simplified example - implement actual external API calls
            external_response = await self._call_external_api(
                failure_context.request_data
            )
            return external_response
        except Exception as e:
            logger.error(f"External API fallback failed: {str(e)}")
            return None
    
    async def _call_external_api(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call external API service"""
        
        # This would implement actual calls to backup services
        # For now, return a placeholder
        return {
            "title": "Simple Recipe",
            "description": "Generated using fallback service",
            "ingredients": request_data.get("ingredients", []),
            "instructions": ["Follow basic cooking principles"]
        }
    
    def _find_best_static_recipe(self, ingredients: List[str]) -> Dict[str, Any]:
        """Find best matching static recipe"""
        
        best_match = None
        best_score = 0
        
        for recipe_key, recipe in self.static_recipes.items():
            if recipe_key == "default":
                continue
            
            # Calculate ingredient overlap
            recipe_ingredients = set(
                ing.lower() for ing in recipe.get("ingredients", [])
            )
            user_ingredients = set(ing.lower() for ing in ingredients)
            
            overlap = len(recipe_ingredients.intersection(user_ingredients))
            score = overlap / len(recipe_ingredients) if recipe_ingredients else 0
            
            if score > best_score:
                best_score = score
                best_match = recipe
        
        return best_match or self.static_recipes.get("default")
    
    def _load_static_recipes(self) -> Dict[str, Dict[str, Any]]:
        """Load static recipe fallbacks"""
        
        return {
            "default": {
                "title": "Simple Pasta",
                "description": "A basic pasta recipe available when AI services are unavailable",
                "prep_time_minutes": 5,
                "cook_time_minutes": 15,
                "total_time_minutes": 20,
                "servings": 4,
                "difficulty": "beginner",
                "ingredients": [
                    {"item": "pasta", "amount": "1", "unit": "lb"},
                    {"item": "olive oil", "amount": "2", "unit": "tbsp"},
                    {"item": "garlic", "amount": "2", "unit": "cloves"},
                    {"item": "salt", "amount": "1", "unit": "tsp"}
                ],
                "instructions": [
                    {"step": 1, "instruction": "Boil water in a large pot"},
                    {"step": 2, "instruction": "Add pasta and cook according to package directions"},
                    {"step": 3, "instruction": "Heat olive oil and add minced garlic"},
                    {"step": 4, "instruction": "Drain pasta and toss with garlic oil"},
                    {"step": 5, "instruction": "Season with salt and serve"}
                ],
                "_fallback_recipe": True
            },
            "chicken": {
                "title": "Basic Grilled Chicken",
                "description": "Simple grilled chicken recipe",
                "prep_time_minutes": 10,
                "cook_time_minutes": 20,
                "ingredients": [
                    {"item": "chicken breast", "amount": "4", "unit": "pieces"},
                    {"item": "olive oil", "amount": "2", "unit": "tbsp"},
                    {"item": "salt", "amount": "1", "unit": "tsp"},
                    {"item": "pepper", "amount": "1/2", "unit": "tsp"}
                ],
                "instructions": [
                    {"step": 1, "instruction": "Preheat grill to medium-high heat"},
                    {"step": 2, "instruction": "Season chicken with salt and pepper"},
                    {"step": 3, "instruction": "Brush with olive oil"},
                    {"step": 4, "instruction": "Grill for 6-8 minutes per side"},
                    {"step": 5, "instruction": "Rest for 5 minutes before serving"}
                ],
                "_fallback_recipe": True
            }
        }
    
    async def _track_success(self, model_name: str, attempts: int):
        """Track successful execution metrics"""
        
        # This would integrate with monitoring system
        logger.info(f"Success for {model_name} after {attempts} attempts")
    
    async def _track_fallback_success(self, fallback_level: FallbackLevel):
        """Track fallback strategy success"""
        
        logger.info(f"Fallback success: {fallback_level.value}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health and fallback statistics"""
        
        recent_failures = [
            f for f in self.failure_history
            if f.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        failure_by_type = {}
        for failure in recent_failures:
            failure_type = failure.failure_type.value
            failure_by_type[failure_type] = failure_by_type.get(failure_type, 0) + 1
        
        circuit_breaker_status = {}
        for model, cb in self.circuit_breakers.items():
            circuit_breaker_status[model] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "recent_failures_count": len(recent_failures),
            "failure_breakdown": failure_by_type,
            "circuit_breakers": circuit_breaker_status,
            "fallback_config": {
                "max_retries": self.config.max_retries,
                "circuit_breaker_threshold": self.config.circuit_breaker_threshold,
                "fallback_levels_enabled": {
                    "cache": self.config.enable_cache_fallback,
                    "model_downgrade": self.config.enable_model_downgrade,
                    "template": self.config.enable_template_fallback,
                    "external": self.config.enable_external_fallback
                }
            }
        }

class RecipeTemplateEngine:
    """Template-based recipe generation for fallback scenarios"""
    
    def __init__(self):
        self.recipe_templates = self._load_recipe_templates()
        self.cooking_methods = self._load_cooking_methods()
    
    async def generate_recipe_from_template(
        self,
        ingredients: List[str],
        dietary_preferences: List[str] = None,
        cuisine_type: str = None,
        cooking_time: int = None
    ) -> Dict[str, Any]:
        """Generate recipe using templates and rules"""
        
        dietary_preferences = dietary_preferences or []
        
        # Select appropriate template
        template = self._select_template(ingredients, dietary_preferences, cuisine_type)
        
        # Generate recipe from template
        recipe = {
            "title": self._generate_title(ingredients, template),
            "description": template["description_template"].format(
                main_ingredient=ingredients[0] if ingredients else "mixed ingredients"
            ),
            "prep_time_minutes": template["prep_time"],
            "cook_time_minutes": template["cook_time"],
            "total_time_minutes": template["prep_time"] + template["cook_time"],
            "servings": 4,
            "difficulty": template["difficulty"],
            "ingredients": self._format_ingredients(ingredients),
            "instructions": self._generate_instructions(ingredients, template),
            "nutrition_per_serving": self._estimate_nutrition(ingredients),
            "_template_generated": True
        }
        
        return recipe
    
    async def generate_meal_plan_from_template(
        self,
        days: int,
        dietary_preferences: List[str] = None
    ) -> Dict[str, Any]:
        """Generate meal plan using templates"""
        
        meal_templates = self._get_meal_templates(dietary_preferences or [])
        
        meal_plan = {
            "week_overview": {
                "total_recipes": days * 3,
                "prep_strategy": "Template-based meal planning",
                "nutritional_balance": "Balanced variety of template recipes"
            },
            "days": [],
            "_template_generated": True
        }
        
        for day in range(1, days + 1):
            day_meals = {
                "day": day,
                "date": f"Day {day}",
                "meals": [
                    {
                        "meal_type": "breakfast",
                        "recipe_name": random.choice(meal_templates["breakfast"])["title"],
                        "prep_time_minutes": 15,
                        "calories": 400
                    },
                    {
                        "meal_type": "lunch",
                        "recipe_name": random.choice(meal_templates["lunch"])["title"],
                        "prep_time_minutes": 25,
                        "calories": 500
                    },
                    {
                        "meal_type": "dinner",
                        "recipe_name": random.choice(meal_templates["dinner"])["title"],
                        "prep_time_minutes": 35,
                        "calories": 600
                    }
                ]
            }
            meal_plan["days"].append(day_meals)
        
        return meal_plan
    
    def _select_template(
        self,
        ingredients: List[str],
        dietary_preferences: List[str],
        cuisine_type: str
    ) -> Dict[str, Any]:
        """Select most appropriate template"""
        
        # Simple template selection logic
        if any("chicken" in ing.lower() for ing in ingredients):
            return self.recipe_templates["protein"]
        elif any("pasta" in ing.lower() for ing in ingredients):
            return self.recipe_templates["pasta"]
        elif any("rice" in ing.lower() for ing in ingredients):
            return self.recipe_templates["grain"]
        else:
            return self.recipe_templates["basic"]
    
    def _generate_title(self, ingredients: List[str], template: Dict[str, Any]) -> str:
        """Generate recipe title"""
        
        main_ingredient = ingredients[0] if ingredients else "Mixed"
        return f"{template['title_prefix']} {main_ingredient.title()}"
    
    def _format_ingredients(self, ingredients: List[str]) -> List[Dict[str, Any]]:
        """Format ingredients with basic measurements"""
        
        formatted = []
        for ing in ingredients:
            formatted.append({
                "item": ing,
                "amount": "1",
                "unit": "cup",  # Default unit
                "preparation": "prepared as needed"
            })
        
        return formatted
    
    def _generate_instructions(
        self,
        ingredients: List[str],
        template: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate cooking instructions"""
        
        instructions = []
        for i, instruction_template in enumerate(template["instruction_templates"]):
            instructions.append({
                "step": i + 1,
                "instruction": instruction_template.format(
                    ingredients=", ".join(ingredients)
                ),
                "time_minutes": 5
            })
        
        return instructions
    
    def _estimate_nutrition(self, ingredients: List[str]) -> Dict[str, int]:
        """Provide basic nutrition estimates"""
        
        return {
            "calories": 350,
            "protein_g": 25,
            "carbs_g": 40,
            "fat_g": 15,
            "fiber_g": 5,
            "sugar_g": 8,
            "sodium_mg": 400
        }
    
    def _load_recipe_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load recipe templates"""
        
        return {
            "basic": {
                "title_prefix": "Simple",
                "description_template": "A straightforward recipe featuring {main_ingredient}",
                "prep_time": 15,
                "cook_time": 25,
                "difficulty": "beginner",
                "instruction_templates": [
                    "Prepare all ingredients: {ingredients}",
                    "Heat cooking vessel to appropriate temperature",
                    "Combine and cook ingredients according to basic principles",
                    "Season to taste and serve hot"
                ]
            },
            "protein": {
                "title_prefix": "Grilled",
                "description_template": "Perfectly grilled {main_ingredient} with complementary flavors",
                "prep_time": 10,
                "cook_time": 20,
                "difficulty": "intermediate",
                "instruction_templates": [
                    "Season {ingredients} with salt and pepper",
                    "Preheat grill or pan to medium-high heat",
                    "Cook protein until internal temperature is safe",
                    "Rest briefly and serve with sides"
                ]
            },
            "pasta": {
                "title_prefix": "Classic",
                "description_template": "Traditional pasta dish with {main_ingredient}",
                "prep_time": 5,
                "cook_time": 15,
                "difficulty": "beginner",
                "instruction_templates": [
                    "Bring large pot of salted water to boil",
                    "Cook pasta according to package directions",
                    "Prepare sauce with {ingredients}",
                    "Combine pasta and sauce, serve immediately"
                ]
            },
            "grain": {
                "title_prefix": "Hearty",
                "description_template": "Wholesome grain-based dish with {main_ingredient}",
                "prep_time": 10,
                "cook_time": 30,
                "difficulty": "intermediate",
                "instruction_templates": [
                    "Rinse and prepare grains",
                    "Cook grains with appropriate liquid ratio",
                    "Add {ingredients} during cooking process",
                    "Fluff and season before serving"
                ]
            }
        }
    
    def _load_cooking_methods(self) -> Dict[str, List[str]]:
        """Load cooking method templates"""
        
        return {
            "quick": ["sauté", "stir-fry", "pan-fry"],
            "slow": ["braise", "stew", "slow-cook"],
            "healthy": ["steam", "poach", "grill"],
            "comfort": ["bake", "roast", "casserole"]
        }
    
    def _get_meal_templates(self, dietary_preferences: List[str]) -> Dict[str, List[Dict]]:
        """Get meal templates by type"""
        
        return {
            "breakfast": [
                {"title": "Hearty Morning Bowl"},
                {"title": "Quick Breakfast Plate"},
                {"title": "Energizing Start"}
            ],
            "lunch": [
                {"title": "Satisfying Midday Meal"},
                {"title": "Light Lunch Option"},
                {"title": "Power Lunch Bowl"}
            ],
            "dinner": [
                {"title": "Complete Evening Dinner"},
                {"title": "Family Style Meal"},
                {"title": "Gourmet Night Dish"}
            ]
        }
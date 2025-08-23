"""
Request and Response models for ChefoodAI AI Service
Comprehensive validation and standardized API models
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
import uuid


class ResponseStatus(str, Enum):
    """Standard response statuses"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class ErrorType(str, Enum):
    """Error types for standardized error handling"""
    VALIDATION_ERROR = "validation_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_ERROR = "authentication_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT_ERROR = "timeout_error"
    INTERNAL_ERROR = "internal_error"


class BaseResponse(BaseModel):
    """Base response model for all API responses"""
    status: ResponseStatus
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ErrorResponse(BaseResponse):
    """Standardized error response"""
    status: ResponseStatus = ResponseStatus.ERROR
    error_type: ErrorType
    detail: str
    errors: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: Dict[str, bool] = Field(default_factory=dict)
    uptime: Optional[float] = None


# Recipe Generation Models
class DietaryRestriction(str, Enum):
    """Supported dietary restrictions"""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    LOW_CARB = "low_carb"
    KETO = "keto"
    PALEO = "paleo"
    HALAL = "halal"
    KOSHER = "kosher"


class DifficultyLevel(str, Enum):
    """Recipe difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class CuisineType(str, Enum):
    """Supported cuisine types"""
    ITALIAN = "italian"
    MEXICAN = "mexican"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    INDIAN = "indian"
    FRENCH = "french"
    AMERICAN = "american"
    MEDITERRANEAN = "mediterranean"
    THAI = "thai"
    KOREAN = "korean"
    FUSION = "fusion"


class RecipeGenerationRequest(BaseModel):
    """Request model for recipe generation"""
    prompt: str = Field(..., min_length=3, max_length=1000, description="Recipe generation prompt")
    cuisine: Optional[CuisineType] = None
    dietary_restrictions: List[DietaryRestriction] = Field(default_factory=list)
    servings: int = Field(default=4, ge=1, le=12, description="Number of servings")
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    max_time: Optional[int] = Field(None, ge=10, le=480, description="Maximum cooking time in minutes")
    ingredients: Optional[List[str]] = Field(None, max_items=20)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI model temperature")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError('Prompt must be at least 3 characters')
        return v
    
    @validator('ingredients')
    def validate_ingredients(cls, v):
        if v:
            return [ingredient.strip() for ingredient in v if ingredient.strip()]
        return v


class NutritionInfo(BaseModel):
    """Nutrition information model"""
    calories: Optional[float] = Field(None, ge=0)
    protein: Optional[float] = Field(None, ge=0, description="Protein in grams")
    carbs: Optional[float] = Field(None, ge=0, description="Carbohydrates in grams")
    fat: Optional[float] = Field(None, ge=0, description="Fat in grams")
    fiber: Optional[float] = Field(None, ge=0, description="Fiber in grams")
    sugar: Optional[float] = Field(None, ge=0, description="Sugar in grams")
    sodium: Optional[float] = Field(None, ge=0, description="Sodium in milligrams")


class Recipe(BaseModel):
    """Recipe model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    ingredients: List[str] = Field(..., min_items=1, max_items=50)
    instructions: List[str] = Field(..., min_items=1, max_items=20)
    prep_time: Optional[int] = Field(None, ge=0, description="Preparation time in minutes")
    cook_time: Optional[int] = Field(None, ge=0, description="Cooking time in minutes")
    total_time: Optional[int] = Field(None, ge=0, description="Total time in minutes")
    servings: int = Field(..., ge=1, le=12)
    difficulty: DifficultyLevel
    cuisine: Optional[CuisineType] = None
    dietary_restrictions: List[DietaryRestriction] = Field(default_factory=list)
    nutrition: Optional[NutritionInfo] = None
    tags: List[str] = Field(default_factory=list, max_items=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RecipeGenerationResponse(BaseResponse):
    """Response model for recipe generation"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    recipe: Recipe
    generated_by: str = "ai-service"
    model_used: str
    processing_time: Optional[float] = None


# Image Generation Models
class ImageStyle(str, Enum):
    """Image generation styles"""
    PROFESSIONAL_FOOD_PHOTOGRAPHY = "professional_food_photography"
    RUSTIC = "rustic"
    MODERN = "modern"
    ARTISTIC = "artistic"
    MINIMALIST = "minimalist"


class ImageSize(str, Enum):
    """Supported image sizes"""
    SQUARE_1024 = "1024x1024"
    SQUARE_512 = "512x512"
    LANDSCAPE_1792_1024 = "1792x1024"
    PORTRAIT_1024_1792 = "1024x1792"


class ImageQuality(str, Enum):
    """Image quality options"""
    STANDARD = "standard"
    HD = "hd"


class ImageProvider(str, Enum):
    """Image generation providers"""
    DALLE = "dalle"
    VERTEX = "vertex"


class ImageGenerationRequest(BaseModel):
    """Request model for image generation"""
    prompt: str = Field(..., min_length=3, max_length=1000, description="Image generation prompt")
    recipe_name: Optional[str] = Field(None, max_length=200)
    style: ImageStyle = ImageStyle.PROFESSIONAL_FOOD_PHOTOGRAPHY
    size: ImageSize = ImageSize.SQUARE_1024
    quality: ImageQuality = ImageQuality.HD
    provider: ImageProvider = ImageProvider.DALLE
    
    @validator('prompt')
    def validate_prompt(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError('Prompt must be at least 3 characters')
        return v


class ImageGenerationResponse(BaseResponse):
    """Response model for image generation"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    image_url: HttpUrl
    provider: ImageProvider
    prompt: str
    generated_by: str = "ai-service"
    processing_time: Optional[float] = None


# Nutrition Analysis Models
class Ingredient(BaseModel):
    """Ingredient model for nutrition analysis"""
    name: str = Field(..., min_length=1, max_length=100)
    quantity: float = Field(..., gt=0)
    unit: str = Field(..., min_length=1, max_length=20)
    
    @validator('name')
    def validate_name(cls, v):
        return v.strip()
    
    @validator('unit')
    def validate_unit(cls, v):
        return v.strip().lower()


class NutritionAnalysisRequest(BaseModel):
    """Request model for nutrition analysis"""
    ingredients: List[Ingredient] = Field(..., min_items=1, max_items=50)
    servings: int = Field(default=1, ge=1, le=12)
    recipe_text: Optional[str] = Field(None, max_length=5000)


class DetailedNutritionInfo(NutritionInfo):
    """Extended nutrition information"""
    vitamins: Dict[str, float] = Field(default_factory=dict)
    minerals: Dict[str, float] = Field(default_factory=dict)
    fatty_acids: Dict[str, float] = Field(default_factory=dict)
    amino_acids: Dict[str, float] = Field(default_factory=dict)


class NutritionAnalysisResponse(BaseResponse):
    """Response model for nutrition analysis"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    nutrition: DetailedNutritionInfo
    per_serving: DetailedNutritionInfo
    ingredients_analyzed: int
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: Optional[float] = None


# Meal Plan Models
class MealType(str, Enum):
    """Meal types"""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class NutritionalGoal(BaseModel):
    """Nutritional goals for meal planning"""
    calories: Optional[float] = Field(None, ge=800, le=5000)
    protein_percent: Optional[float] = Field(None, ge=10, le=40)
    carbs_percent: Optional[float] = Field(None, ge=20, le=70)
    fat_percent: Optional[float] = Field(None, ge=15, le=50)
    
    @validator('protein_percent', 'carbs_percent', 'fat_percent')
    def validate_percentages(cls, v, values):
        # Validate that percentages don't exceed 100% when combined
        return v


class MealPlanOptimizationRequest(BaseModel):
    """Request model for meal plan optimization"""
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    dietary_restrictions: List[DietaryRestriction] = Field(default_factory=list)
    nutritional_goals: NutritionalGoal
    days: int = Field(default=7, ge=1, le=14)
    meals_per_day: int = Field(default=3, ge=2, le=6)
    budget: Optional[float] = Field(None, gt=0, description="Budget per day")
    exclude_ingredients: List[str] = Field(default_factory=list, max_items=20)
    
    @validator('exclude_ingredients')
    def validate_exclude_ingredients(cls, v):
        return [ingredient.strip().lower() for ingredient in v if ingredient.strip()]


class Meal(BaseModel):
    """Individual meal model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MealType
    recipe: Recipe
    scheduled_time: Optional[str] = None  # ISO time format


class DayPlan(BaseModel):
    """Daily meal plan"""
    date: str  # ISO date format
    meals: List[Meal]
    total_nutrition: NutritionInfo
    estimated_cost: Optional[float] = None


class MealPlan(BaseModel):
    """Complete meal plan"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    days: List[DayPlan]
    total_nutrition: NutritionInfo
    estimated_total_cost: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MealPlanOptimizationResponse(BaseResponse):
    """Response model for meal plan optimization"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    meal_plan: MealPlan
    optimization_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: Optional[float] = None


# Shopping List Models
class ShoppingItem(BaseModel):
    """Shopping list item"""
    name: str = Field(..., min_length=1, max_length=100)
    quantity: float = Field(..., gt=0)
    unit: str = Field(..., min_length=1, max_length=20)
    category: Optional[str] = None
    estimated_price: Optional[float] = Field(None, ge=0)
    priority: int = Field(default=1, ge=1, le=5)
    
    @validator('name')
    def validate_name(cls, v):
        return v.strip()


class OptimizeFor(str, Enum):
    """Shopping optimization criteria"""
    EFFICIENCY = "efficiency"
    COST = "cost"
    HEALTH = "health"
    TIME = "time"


class ShoppingListOptimizationRequest(BaseModel):
    """Request model for shopping list optimization"""
    items: List[ShoppingItem] = Field(..., min_items=1, max_items=100)
    store_layout: Optional[str] = Field(None, max_length=50)
    budget: Optional[float] = Field(None, gt=0)
    optimize_for: OptimizeFor = OptimizeFor.EFFICIENCY
    max_stores: int = Field(default=1, ge=1, le=5)


class OptimizedShoppingList(BaseModel):
    """Optimized shopping list"""
    items: List[ShoppingItem]
    total_estimated_cost: Optional[float] = None
    estimated_time: Optional[int] = None  # minutes
    optimization_score: float = Field(..., ge=0.0, le=1.0)
    recommendations: List[str] = Field(default_factory=list)


class ShoppingListOptimizationResponse(BaseResponse):
    """Response model for shopping list optimization"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    optimized_list: OptimizedShoppingList
    processing_time: Optional[float] = None
"""
API Routes Package for ChefoodAI AI Service
Organized route modules for different service areas
"""

from .health import router as health_router
from .recipes import router as recipes_router
from .images import router as images_router
from .nutrition import router as nutrition_router
from .meal_plans import router as meal_plans_router
from .shopping import router as shopping_router

__all__ = [
    "health_router",
    "recipes_router", 
    "images_router",
    "nutrition_router",
    "meal_plans_router",
    "shopping_router"
]
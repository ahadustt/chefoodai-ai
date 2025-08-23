"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from httpx import AsyncClient
import os

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["CORS_ORIGINS"] = "http://localhost:3000,http://localhost:8000"

from config import get_settings
from api import create_app


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Test settings configuration"""
    return get_settings()


@pytest.fixture(scope="session")
def app():
    """Create test FastAPI application"""
    return create_app()


@pytest.fixture(scope="session")
def client(app) -> TestClient:
    """Create test client"""
    return TestClient(app)


@pytest.fixture(scope="session")
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_recipe_request():
    """Sample recipe generation request"""
    return {
        "prompt": "Make a healthy pasta dish",
        "cuisine": "italian",
        "dietary_restrictions": ["vegetarian"],
        "servings": 4,
        "difficulty": "medium",
        "max_time": 60
    }


@pytest.fixture
def sample_nutrition_request():
    """Sample nutrition analysis request"""
    return {
        "ingredients": [
            {"name": "chicken breast", "quantity": 200, "unit": "g"},
            {"name": "broccoli", "quantity": 150, "unit": "g"},
            {"name": "brown rice", "quantity": 100, "unit": "g"}
        ],
        "servings": 2
    }


@pytest.fixture
def sample_image_request():
    """Sample image generation request"""
    return {
        "prompt": "Beautiful Italian pasta dish with fresh herbs",
        "style": "professional_food_photography",
        "size": "1024x1024",
        "quality": "hd",
        "provider": "dalle"
    }


@pytest.fixture
def sample_meal_plan_request():
    """Sample meal plan optimization request"""
    return {
        "user_preferences": {"cuisine": "mediterranean"},
        "dietary_restrictions": ["vegetarian"],
        "nutritional_goals": {
            "calories": 2000,
            "protein_percent": 20,
            "carbs_percent": 50,
            "fat_percent": 30
        },
        "days": 7,
        "meals_per_day": 3,
        "budget": 100.0
    }


@pytest.fixture
def sample_shopping_request():
    """Sample shopping list optimization request"""
    return {
        "items": [
            {"name": "Tomatoes", "quantity": 2, "unit": "lbs", "category": "Produce"},
            {"name": "Chicken breast", "quantity": 1, "unit": "lb", "category": "Meat"},
            {"name": "Rice", "quantity": 1, "unit": "bag", "category": "Pantry"}
        ],
        "store_layout": "standard_grocery",
        "budget": 50.0,
        "optimize_for": "efficiency"
    }
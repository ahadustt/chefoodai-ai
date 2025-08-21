"""
Google Vertex AI Integration for ChefoodAI
Handles recipe generation, meal planning, and multi-modal capabilities
"""

import os
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import logging
import vertexai

# Optional imports for caching
try:
    from redis import asyncio as redis_async
    REDIS_ASYNC_AVAILABLE = True
except ImportError:
    redis_async = None
    REDIS_ASYNC_AVAILABLE = False
import uuid
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    GenerationConfig
)

# Import SafetySettings with fallback for version compatibility
try:
    from vertexai.generative_models import SafetySettings, HarmCategory, HarmBlockThreshold
except ImportError:
    # Fallback for older Vertex AI versions
    SafetySettings = None
    HarmCategory = None
    HarmBlockThreshold = None
from vertexai.preview.vision_models import ImageGenerationModel
import google.cloud.storage as storage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ModelVersion(Enum):
    """Available Gemini model versions"""
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_0_PRO = "gemini-1.0-pro"


class InferenceMode(Enum):
    """Inference patterns"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"


class RecipeRequest(BaseModel):
    """Recipe generation request model"""
    ingredients: List[str]
    dietary_preferences: List[str] = Field(default_factory=list)
    cuisine_type: Optional[str] = None
    cooking_time_minutes: Optional[int] = None
    skill_level: Optional[str] = "intermediate"
    servings: int = 4
    exclude_ingredients: List[str] = Field(default_factory=list)
    cooking_methods: List[str] = Field(default_factory=list)
    meal_type: Optional[str] = None
    include_images: bool = False
    generate_ingredient_images: bool = False

class MealPlanRequest(BaseModel):
    """Meal planning request model"""
    days: int = 7
    meals_per_day: int = 3
    dietary_preferences: List[str] = Field(default_factory=list)
    calorie_target: Optional[int] = None
    budget_per_week: Optional[float] = None
    family_size: int = 4
    prep_time_preference: str = "moderate"  # quick, moderate, elaborate


class VertexAIService:
    """Main service for Vertex AI integration"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.redis_client = None
        self.models = {}
        self.storage_bucket = os.getenv("STORAGE_BUCKET", f"{project_id}-recipe-images")
        self._initialize_vertex_ai()
        self._initialize_storage()
        
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI with project settings"""
        vertexai.init(project=self.project_id, location=self.location)
        
        # Initialize different model versions
        self.models[ModelVersion.GEMINI_1_5_PRO] = GenerativeModel(
            ModelVersion.GEMINI_1_5_PRO.value
        )
        self.models[ModelVersion.GEMINI_1_5_FLASH] = GenerativeModel(
            ModelVersion.GEMINI_1_5_FLASH.value
        )
        
        # Initialize image generation model
        try:
            self.imagen_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info("Imagen model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Imagen model: {e}")
            self.imagen_model = None
            
    def _initialize_storage(self):
        """Initialize Google Cloud Storage for image storage"""
        try:
            self.storage_client = storage.Client(project=self.project_id)
            # Ensure bucket exists
            try:
                self.bucket = self.storage_client.bucket(self.storage_bucket)
                self.bucket.reload()  # Check if bucket exists
            except Exception:
                # Create bucket if it doesn't exist
                self.bucket = self.storage_client.create_bucket(
                    self.storage_bucket,
                    location=self.location
                )
                logger.info(f"Created storage bucket: {self.storage_bucket}")
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            self.storage_client = None
            self.bucket = None


    async def initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis for caching (optional)"""
        if not AIOREDIS_AVAILABLE:
            logger.warning("aioredis not available - caching disabled")
            self.redis_client = None
            return
        
        try:
            self.redis_client = await aioredis.create_redis_pool(redis_url)
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None
        
    def get_safety_settings(self) -> List[Any]:
        """Configure safety settings for content moderation with version compatibility"""
        if SafetySettings is None or HarmCategory is None:
            # Return empty list for older Vertex AI versions
            return []
        
        return [
            SafetySettings(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            SafetySettings(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            SafetySettings(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            SafetySettings(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
        ]
    
    def get_generation_config(self, mode: InferenceMode) -> GenerationConfig:
        """Get generation config based on inference mode"""
        base_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        if mode == InferenceMode.BATCH:
            base_config["temperature"] = 0.5  # More consistent for batch
        elif mode == InferenceMode.STREAMING:
            base_config["max_output_tokens"] = 1024  # Smaller chunks for streaming
            
        return GenerationConfig(**base_config)
    
    async def generate_recipe(
        self, 
        request: RecipeRequest,
        model_version: ModelVersion = ModelVersion.GEMINI_1_5_PRO,
        use_cache: bool = True,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a recipe based on user preferences"""
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key("recipe", request.dict())
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for recipe request: {cache_key}")
                return cached_result
        
        # Select prompt template based on experiment
        prompt = self._get_recipe_prompt(request, experiment_id)
        
        try:
            # Generate recipe
            model = self.models[model_version]
            response = await self._generate_with_retry(
                model=model,
                prompt=prompt,
                generation_config=self.get_generation_config(InferenceMode.REAL_TIME),
                safety_settings=self.get_safety_settings()
            )
            
            # Parse and validate response
            recipe_data = self._parse_recipe_response(response.text)
            
            # Generate images if requested
            if request.include_images and recipe_data:
                recipe_data["image_url"] = await self.generate_recipe_image(
                    recipe_data["title"],
                    recipe_data.get("description", "")
                )
            
            # Generate ingredient images if requested
            if request.generate_ingredient_images and recipe_data:
                ingredients = recipe_data.get("ingredients", [])
                if isinstance(ingredients, list) and ingredients:
                    # Extract ingredient names from structured data
                    ingredient_names = []
                    for ingredient in ingredients:
                        if isinstance(ingredient, dict):
                            ingredient_names.append(ingredient.get("item", ""))
                        else:
                            ingredient_names.append(str(ingredient))
                    
                    ingredient_images = await self.generate_ingredient_images(ingredient_names)
                    recipe_data["ingredient_images"] = ingredient_images
            
            # Cache the result
            if use_cache and recipe_data:
                await self._set_cache(cache_key, recipe_data, ttl=3600)
            
            # Log for A/B testing
            if experiment_id:
                await self._log_experiment_result(
                    experiment_id,
                    request.dict(),
                    recipe_data,
                    model_version.value
                )
            
            return recipe_data
            
        except Exception as e:
            logger.error(f"Recipe generation failed: {str(e)}")
            # Fallback to simpler model
            if model_version == ModelVersion.GEMINI_1_5_PRO:
                return await self.generate_recipe(
                    request,
                    ModelVersion.GEMINI_1_5_FLASH,
                    use_cache=False
                )
            raise
    
    def _get_recipe_prompt(self, request: RecipeRequest, experiment_id: Optional[str] = None) -> str:
        """Get recipe generation prompt based on experiment"""
        
        # Base prompt template
        base_template = """You are ChefoodAI, an expert culinary assistant. Generate a detailed recipe based on the following requirements:

**Available Ingredients:**
{ingredients}

**Dietary Preferences:** {dietary_preferences}
**Cuisine Type:** {cuisine_type}
**Cooking Time:** {cooking_time} minutes
**Skill Level:** {skill_level}
**Servings:** {servings}
**Excluded Ingredients:** {exclude_ingredients}
**Preferred Cooking Methods:** {cooking_methods}
**Meal Type:** {meal_type}

Please provide a recipe in the following JSON format:
{{
    "title": "Recipe Name",
    "description": "Brief description",
    "prep_time_minutes": 15,
    "cook_time_minutes": 30,
    "total_time_minutes": 45,
    "servings": 4,
    "difficulty": "intermediate",
    "cuisine": "cuisine type",
    "meal_type": "dinner",
    "dietary_info": ["vegetarian", "gluten-free"],
    "ingredients": [
        {{
            "item": "ingredient name",
            "amount": "2",
            "unit": "cups",
            "preparation": "diced",
            "substitutions": ["alternative 1", "alternative 2"]
        }}
    ],
    "equipment": ["pot", "pan", "oven"],
    "instructions": [
        {{
            "step": 1,
            "instruction": "Detailed instruction",
            "time_minutes": 5,
            "tips": "Helpful tip for this step"
        }}
    ],
    "nutrition_per_serving": {{
        "calories": 350,
        "protein_g": 25,
        "carbs_g": 40,
        "fat_g": 15,
        "fiber_g": 8,
        "sugar_g": 5,
        "sodium_mg": 450
    }},
    "tips": ["tip 1", "tip 2"],
    "variations": ["variation 1", "variation 2"],
    "storage": "Storage instructions",
    "tags": ["healthy", "quick", "family-friendly"]
}}

Be creative but practical. Ensure all measurements are precise and instructions are clear."""

        # A/B test different prompt variations
        if experiment_id == "detailed_context":
            base_template += "\n\nAdditional context: Consider seasonal ingredients and local availability. Provide extra detail on technique and flavor development."
        elif experiment_id == "conversational":
            base_template = base_template.replace(
                "You are ChefoodAI, an expert culinary assistant.",
                "Hi! I'm ChefoodAI, your friendly kitchen companion. Let me help you create something delicious!"
            )
        elif experiment_id == "minimal":
            # Shorter, more concise prompt
            base_template = """Generate a recipe with these ingredients: {ingredients}
Dietary: {dietary_preferences}
Time: {cooking_time} minutes
Output JSON with: title, ingredients (with amounts), instructions, nutrition."""

        return base_template.format(
            ingredients=", ".join(request.ingredients),
            dietary_preferences=", ".join(request.dietary_preferences) or "None specified",
            cuisine_type=request.cuisine_type or "Any",
            cooking_time=request.cooking_time_minutes or "Any",
            skill_level=request.skill_level,
            servings=request.servings,
            exclude_ingredients=", ".join(request.exclude_ingredients) or "None",
            cooking_methods=", ".join(request.cooking_methods) or "Any",
            meal_type=request.meal_type or "Any"
        )
    
    async def generate_meal_plan(
        self,
        request: MealPlanRequest,
        model_version: ModelVersion = ModelVersion.GEMINI_1_5_PRO
    ) -> Dict[str, Any]:
        """Generate a complete meal plan"""
        
        # Use context window optimization for longer output
        prompt = self._get_meal_plan_prompt(request)
        
        # For meal plans, we need larger context window
        generation_config = GenerationConfig(
            temperature=0.6,
            top_p=0.9,
            top_k=30,
            max_output_tokens=8192,  # Larger output for complete meal plan
        )
        
        try:
            model = self.models[model_version]
            response = await self._generate_with_retry(
                model=model,
                prompt=prompt,
                generation_config=generation_config,
                safety_settings=self.get_safety_settings()
            )
            
            meal_plan = self._parse_meal_plan_response(response.text)
            
            # Batch generate shopping list
            if meal_plan:
                meal_plan["shopping_list"] = await self.generate_shopping_list(meal_plan)
            
            return meal_plan
            
        except Exception as e:
            logger.error(f"Meal plan generation failed: {str(e)}")
            raise
    
    def _get_meal_plan_prompt(self, request: MealPlanRequest) -> str:
        """Generate meal planning prompt with context optimization"""
        
        return f"""You are ChefoodAI, planning a {request.days}-day meal plan for a family of {request.family_size}.

**Requirements:**
- Meals per day: {request.meals_per_day}
- Dietary preferences: {', '.join(request.dietary_preferences) or 'None specified'}
- Daily calorie target: {request.calorie_target or 'Not specified'}
- Weekly budget: ${request.budget_per_week or 'Not specified'}
- Prep time preference: {request.prep_time_preference}

**Context for optimization:**
1. Reuse ingredients across meals to minimize waste and cost
2. Balance nutrition across the week
3. Include variety to prevent meal fatigue
4. Consider batch cooking opportunities
5. Plan for leftovers strategically

Generate a meal plan in the following JSON format:
{{
    "week_overview": {{
        "total_recipes": 15,
        "estimated_cost": 120.50,
        "prep_strategy": "Batch cook on Sunday and Wednesday",
        "nutritional_balance": "Well-balanced with emphasis on whole foods"
    }},
    "days": [
        {{
            "day": 1,
            "date": "Monday",
            "meals": [
                {{
                    "meal_type": "breakfast",
                    "recipe_id": "unique_id",
                    "recipe_name": "Recipe Name",
                    "prep_time_minutes": 10,
                    "calories": 400,
                    "key_ingredients": ["ingredient1", "ingredient2"],
                    "leftovers_from": null,
                    "makes_leftovers_for": "Tuesday lunch"
                }}
            ],
            "daily_nutrition": {{
                "calories": 2000,
                "protein_g": 80,
                "carbs_g": 250,
                "fat_g": 70
            }}
        }}
    ],
    "batch_cooking": [
        {{
            "day": "Sunday",
            "recipes": ["recipe1", "recipe2"],
            "time_required_minutes": 120,
            "items_prepped": ["chopped vegetables", "cooked grains", "marinated proteins"]
        }}
    ],
    "shopping_categories": {{
        "proteins": ["chicken breast 2 lbs", "tofu 1 block"],
        "produce": ["spinach 2 bunches", "tomatoes 6"],
        "pantry": ["rice 2 cups", "olive oil"],
        "dairy": ["milk 1 gallon", "yogurt 32 oz"]
    }}
}}

Ensure variety, nutritional balance, and practical cooking flow throughout the week."""
    
    async def generate_recipe_image(self, recipe_title: str, description: str) -> Optional[str]:
        """Generate recipe image using Google's Imagen API"""
        
        if not self.imagen_model or not self.bucket:
            logger.warning("Image generation not available - Imagen model or storage not initialized")
            return None
            
        try:
            # Create optimized prompt for food photography
            image_prompt = self._create_food_prompt(recipe_title, description, "recipe")
            
            # Generate image using Imagen
            logger.info(f"Generating recipe image with prompt: {image_prompt}")
            images = self.imagen_model.generate_images(
                prompt=image_prompt,
                number_of_images=1,
                language="en",
                aspect_ratio="1:1",
                safety_filter_level="block_some",
                person_generation="dont_allow"
            )
            
            if not images:
                logger.error("No images generated")
                return None
                
            # Upload to Cloud Storage
            image_url = await self._upload_image_to_storage(
                images[0]._pil_image, 
                f"recipes/{uuid.uuid4()}.jpg",
                "recipe_image"
            )
            
            logger.info(f"Successfully generated and uploaded recipe image: {image_url}")
            return image_url
            
        except Exception as e:
            logger.error(f"Failed to generate recipe image: {e}")
            return None
    
    async def generate_ingredient_images(self, ingredients: List[str]) -> Dict[str, str]:
        """Generate images for individual ingredients"""
        
        if not self.imagen_model or not self.bucket:
            logger.warning("Image generation not available")
            return {}
            
        ingredient_images = {}
        
        for ingredient in ingredients[:5]:  # Limit to first 5 ingredients to avoid costs
            try:
                # Clean ingredient name (remove quantities, etc.)
                clean_ingredient = self._clean_ingredient_name(ingredient)
                
                # Create ingredient-specific prompt
                image_prompt = self._create_food_prompt(clean_ingredient, "", "ingredient")
                
                logger.info(f"Generating ingredient image for: {clean_ingredient}")
                images = self.imagen_model.generate_images(
                    prompt=image_prompt,
                    number_of_images=1,
                    language="en",
                    aspect_ratio="1:1",
                    safety_filter_level="block_some",
                    person_generation="dont_allow"
                )
                
                if images:
                    image_url = await self._upload_image_to_storage(
                        images[0]._pil_image,
                        f"ingredients/{uuid.uuid4()}.jpg",
                        "ingredient_image"
                    )
                    ingredient_images[clean_ingredient] = image_url
                    
            except Exception as e:
                logger.error(f"Failed to generate image for ingredient {ingredient}: {e}")
                continue
                
        return ingredient_images
    
    def _create_food_prompt(self, item_name: str, description: str, image_type: str) -> str:
        """Create optimized prompts for food photography"""
        
        if image_type == "recipe":
            return (
                f"Professional food photography of {item_name}. "
                f"{description}. "
                f"Beautifully plated dish, appetizing presentation, natural lighting, "
                f"shallow depth of field, garnished, restaurant quality, high resolution, "
                f"food styling, clean white background"
            )
        elif image_type == "ingredient":
            return (
                f"Professional product photography of fresh {item_name}. "
                f"Clean, isolated on white background, natural lighting, "
                f"high quality, sharp focus, food photography style, "
                f"vibrant colors, fresh appearance"
            )
        else:
            return f"Professional food photography of {item_name}. High quality, natural lighting."
    
    def _clean_ingredient_name(self, ingredient: str) -> str:
        """Extract clean ingredient name from recipe ingredient line"""
        import re
        
        # Remove quantities, measurements, and preparation instructions
        # Examples: "2 cups diced tomatoes" -> "tomatoes"
        #          "1 lb ground beef" -> "beef"
        
        # Remove common measurements and quantities
        ingredient = re.sub(r'\b\d+(?:\.\d+)?(?:\s*[-/]\s*\d+(?:\.\d+)?)?\s*', '', ingredient)
        ingredient = re.sub(r'\b(?:cups?|tbsp|tablespoons?|tsp|teaspoons?|oz|ounces?|lbs?|pounds?|grams?|kg|ml|liters?)\b', '', ingredient, flags=re.IGNORECASE)
        
        # Remove preparation instructions
        ingredient = re.sub(r'\b(?:diced|chopped|sliced|minced|grated|fresh|dried|cooked|raw|organic|large|small|medium)\b', '', ingredient, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        ingredient = re.sub(r'[,()]+', ' ', ingredient)
        ingredient = ' '.join(ingredient.split())  # Normalize whitespace
        
        return ingredient.strip().lower()
    
    async def _upload_image_to_storage(self, pil_image, blob_name: str, content_type: str) -> str:
        """Upload PIL image to Google Cloud Storage and return public URL"""
        
        try:
            # Convert PIL image to bytes
            import io
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Upload to Cloud Storage
            blob = self.bucket.blob(blob_name)
            blob.metadata = {"content_type": content_type}
            blob.upload_from_string(img_byte_arr, content_type="image/jpeg")
            
            # Make blob publicly readable
            blob.make_public()
            
            return blob.public_url
            
        except Exception as e:
            logger.error(f"Failed to upload image to storage: {e}")
            raise
    
    async def analyze_recipe_image(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze uploaded recipe image using multi-modal Gemini"""
        
        try:
            model = self.models[ModelVersion.GEMINI_1_5_PRO]
            
            # Create image part
            image_part = Part.from_data(
                data=image_data,
                mime_type="image/jpeg"
            )
            
            prompt = """Analyze this food image and provide:
1. Identified dish name
2. Visible ingredients
3. Cooking method used
4. Estimated cooking difficulty
5. Cuisine type
6. Nutritional assessment (healthy/moderate/indulgent)
7. Suggested recipe modifications for common dietary restrictions

Format as JSON."""
            
            response = await self._generate_with_retry(
                model=model,
                prompt=[prompt, image_part],
                generation_config=self.get_generation_config(InferenceMode.REAL_TIME),
                safety_settings=self.get_safety_settings()
            )
            
            return json.loads(response.text)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            raise
    
    async def batch_generate_recipes(
        self,
        requests: List[RecipeRequest],
        max_batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Batch process multiple recipe requests for efficiency"""
        
        results = []
        
        # Process in batches
        for i in range(0, len(requests), max_batch_size):
            batch = requests[i:i + max_batch_size]
            
            # Create batch prompt
            batch_prompt = self._create_batch_prompt(batch)
            
            try:
                model = self.models[ModelVersion.GEMINI_1_5_FLASH]  # Use faster model for batch
                response = await self._generate_with_retry(
                    model=model,
                    prompt=batch_prompt,
                    generation_config=self.get_generation_config(InferenceMode.BATCH),
                    safety_settings=self.get_safety_settings()
                )
                
                batch_results = self._parse_batch_response(response.text)
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Batch generation failed: {str(e)}")
                # Fallback to individual generation
                for request in batch:
                    try:
                        result = await self.generate_recipe(request, use_cache=True)
                        results.append(result)
                    except:
                        results.append(None)
        
        return results
    
    def _create_batch_prompt(self, requests: List[RecipeRequest]) -> str:
        """Create optimized prompt for batch processing"""
        
        prompt = "Generate recipes for the following requests. Return as a JSON array:\n\n"
        
        for idx, request in enumerate(requests):
            prompt += f"Request {idx + 1}:\n"
            prompt += f"- Ingredients: {', '.join(request.ingredients)}\n"
            prompt += f"- Dietary: {', '.join(request.dietary_preferences)}\n"
            prompt += f"- Time: {request.cooking_time_minutes} minutes\n\n"
        
        prompt += "\nReturn format: [{recipe1}, {recipe2}, ...] with full recipe details for each."
        
        return prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _generate_with_retry(
        self,
        model: GenerativeModel,
        prompt: Union[str, List],
        generation_config: GenerationConfig,
        safety_settings: List[Any]
    ) -> Any:
        """Generate content with retry logic for resilience"""
        
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Check for safety blocks
            if response.candidates[0].finish_reason.name == "SAFETY":
                logger.warning("Response blocked by safety filters")
                raise ValueError("Content blocked by safety filters")
            
            return response
            
        except Exception as e:
            logger.error(f"Generation attempt failed: {str(e)}")
            raise
    
    def _generate_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate cache key from request data"""
        
        # Sort dict for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        hash_digest = hashlib.md5(sorted_data.encode()).hexdigest()
        return f"chefood:{prefix}:{hash_digest}"
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from Redis cache"""
        
        if not self.redis_client:
            return None
            
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
        
        return None
    
    async def _set_cache(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set data in Redis cache with TTL"""
        
        if not self.redis_client:
            return
            
        try:
            await self.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    def _parse_recipe_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate recipe response"""
        
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            recipe_data = json.loads(json_text)
            
            # Validate required fields
            required_fields = ["title", "ingredients", "instructions"]
            for field in required_fields:
                if field not in recipe_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return recipe_data
            
        except Exception as e:
            logger.error(f"Failed to parse recipe response: {str(e)}")
            return None
    
    def _parse_meal_plan_response(self, response_text: str) -> Dict[str, Any]:
        """Parse meal plan response"""
        
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            return json.loads(json_text)
            
        except Exception as e:
            logger.error(f"Failed to parse meal plan response: {str(e)}")
            return None
    
    def _parse_batch_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse batch recipe response"""
        
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            return json.loads(json_text)
            
        except Exception as e:
            logger.error(f"Failed to parse batch response: {str(e)}")
            return []
    
    async def generate_shopping_list(self, meal_plan: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate consolidated shopping list from meal plan"""
        
        # This would aggregate ingredients from all recipes
        # For now, return the shopping_categories if available
        return meal_plan.get("shopping_categories", {})
    
    async def _log_experiment_result(
        self,
        experiment_id: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        model_version: str
    ):
        """Log A/B test results for analysis"""
        
        # In production, this would log to BigQuery or similar
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_id": experiment_id,
            "model_version": model_version,
            "request": request,
            "response_quality": len(response.get("instructions", [])) if response else 0,
            "response_time_ms": 0,  # Would be tracked in actual implementation
        }
        
        logger.info(f"Experiment log: {json.dumps(log_entry)}")
    
    async def fine_tune_model(self, training_data_path: str, model_name: str):
        """Fine-tune model with domain-specific cooking data"""
        
        # This would use Vertex AI's fine-tuning capabilities
        # Implementation would involve:
        # 1. Preparing training data in JSONL format
        # 2. Uploading to GCS
        # 3. Creating fine-tuning job
        # 4. Monitoring job progress
        # 5. Deploying fine-tuned model
        
        logger.info(f"Would fine-tune model with data from {training_data_path}")
        
        # Example structure:
        # fine_tuning_job = aiplatform.CustomTrainingJob(
        #     display_name=model_name,
        #     script_path="fine_tune_script.py",
        #     container_uri="gcr.io/vertex-ai/training/pytorch-gpu.1-11:latest",
        # )
        
    async def close(self):
        """Clean up resources"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
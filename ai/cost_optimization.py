"""
Cost optimization strategies for Vertex AI integration
Includes caching, request batching, and intelligent model selection
"""

import json
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
try:
    from redis import asyncio as redis_async
except ImportError:
    redis_async = None
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class CostTier(Enum):
    """Cost tiers for different request types"""
    FREE = "free"           # Basic cached responses
    STANDARD = "standard"   # Gemini 1.5 Flash
    PREMIUM = "premium"     # Gemini 1.5 Pro
    ENTERPRISE = "enterprise"  # Fine-tuned models

@dataclass
class UsageMetrics:
    """Track usage metrics for cost optimization"""
    requests_count: int = 0
    tokens_consumed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0

class CostOptimizer:
    """Main cost optimization service"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        
        # Pricing per 1K tokens (example rates)
        self.pricing = {
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375},
            "gemini-1.5-flash": {"input": 0.00025, "output": 0.00075},
            "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
        }
        
        # Cache TTL strategies
        self.cache_ttl = {
            "recipe_basic": 3600,      # 1 hour for basic recipes
            "recipe_premium": 7200,    # 2 hours for complex recipes
            "meal_plan": 1800,         # 30 minutes for meal plans
            "image_analysis": 86400,   # 24 hours for image analysis
        }
        
        # Batch processing settings
        self.batch_config = {
            "max_batch_size": 10,
            "batch_timeout_seconds": 5,
            "min_batch_size": 3,
        }
        
        self.pending_requests = defaultdict(list)
        self.usage_metrics = defaultdict(UsageMetrics)
    
    async def should_use_cache(self, request_type: str, user_tier: CostTier) -> bool:
        """Determine if request should use cache based on cost optimization"""
        
        # Always use cache for free tier
        if user_tier == CostTier.FREE:
            return True
        
        # Check user's recent usage
        usage = await self.get_user_usage_today(user_tier.value)
        
        # Use cache more aggressively for high-usage users
        if usage["requests_count"] > 100:
            return True
        
        # Cache basic recipe requests more often
        if request_type == "recipe_basic":
            return True
        
        return False
    
    async def select_optimal_model(
        self,
        request_complexity: str,
        user_tier: CostTier,
        budget_remaining: float
    ) -> Tuple[str, bool]:
        """Select the most cost-effective model for the request"""
        
        # Free tier always uses cache or cheapest model
        if user_tier == CostTier.FREE:
            return "gemini-1.5-flash", True  # Use cache
        
        # Enterprise tier can use any model
        if user_tier == CostTier.ENTERPRISE:
            if request_complexity == "high":
                return "gemini-1.5-pro", False
            return "gemini-1.5-flash", False
        
        # Standard/Premium tier optimization
        estimated_cost = self.estimate_request_cost(request_complexity)
        
        if budget_remaining < estimated_cost["gemini-1.5-pro"]:
            if budget_remaining >= estimated_cost["gemini-1.5-flash"]:
                return "gemini-1.5-flash", False
            else:
                return "gemini-1.5-flash", True  # Force cache
        
        # Use Pro model for complex requests if budget allows
        if request_complexity == "high" and user_tier == CostTier.PREMIUM:
            return "gemini-1.5-pro", False
        
        return "gemini-1.5-flash", False
    
    def estimate_request_cost(self, complexity: str) -> Dict[str, float]:
        """Estimate cost for different models based on request complexity"""
        
        # Token estimates based on complexity
        token_estimates = {
            "low": {"input": 200, "output": 500},
            "medium": {"input": 500, "output": 1200},
            "high": {"input": 1000, "output": 2500},
        }
        
        tokens = token_estimates.get(complexity, token_estimates["medium"])
        costs = {}
        
        for model, pricing in self.pricing.items():
            input_cost = (tokens["input"] / 1000) * pricing["input"]
            output_cost = (tokens["output"] / 1000) * pricing["output"]
            costs[model] = input_cost + output_cost
        
        return costs
    
    async def add_to_batch_queue(
        self,
        request_id: str,
        request_data: Dict[str, Any],
        priority: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Add request to batch queue and return batch if ready"""
        
        batch_key = f"batch:{request_data.get('type', 'default')}"
        
        # Add to pending requests
        self.pending_requests[batch_key].append({
            "id": request_id,
            "data": request_data,
            "priority": priority,
            "timestamp": datetime.utcnow()
        })
        
        # Check if batch is ready
        pending = self.pending_requests[batch_key]
        
        # Sort by priority and timestamp
        pending.sort(key=lambda x: (x["priority"], x["timestamp"]))
        
        # Return batch if conditions are met
        if (len(pending) >= self.batch_config["max_batch_size"] or
            (len(pending) >= self.batch_config["min_batch_size"] and
             self._batch_timeout_reached(pending[0]["timestamp"]))):
            
            batch = pending[:self.batch_config["max_batch_size"]]
            self.pending_requests[batch_key] = pending[self.batch_config["max_batch_size"]:]
            
            return batch
        
        return None
    
    def _batch_timeout_reached(self, first_request_time: datetime) -> bool:
        """Check if batch timeout has been reached"""
        timeout = timedelta(seconds=self.batch_config["batch_timeout_seconds"])
        return datetime.utcnow() - first_request_time >= timeout
    
    async def get_cache_key_with_strategy(
        self,
        base_key: str,
        request_data: Dict[str, Any],
        strategy: str = "exact"
    ) -> str:
        """Generate cache key with different matching strategies"""
        
        if strategy == "exact":
            # Exact match - most specific
            data_hash = hashlib.md5(
                json.dumps(request_data, sort_keys=True).encode()
            ).hexdigest()
            return f"{base_key}:exact:{data_hash}"
        
        elif strategy == "similar":
            # Similar ingredients, ignore order and minor differences
            ingredients = sorted(request_data.get("ingredients", []))
            dietary = sorted(request_data.get("dietary_preferences", []))
            
            similar_data = {
                "ingredients": ingredients,
                "dietary": dietary,
                "cuisine": request_data.get("cuisine_type"),
                "time_range": self._normalize_time_range(
                    request_data.get("cooking_time_minutes")
                )
            }
            
            data_hash = hashlib.md5(
                json.dumps(similar_data, sort_keys=True).encode()
            ).hexdigest()
            return f"{base_key}:similar:{data_hash}"
        
        elif strategy == "loose":
            # Loose match - just main ingredients and dietary restrictions
            main_ingredients = sorted(request_data.get("ingredients", []))[:5]
            main_dietary = sorted(request_data.get("dietary_preferences", []))
            
            loose_data = {
                "main_ingredients": main_ingredients,
                "dietary": main_dietary
            }
            
            data_hash = hashlib.md5(
                json.dumps(loose_data, sort_keys=True).encode()
            ).hexdigest()
            return f"{base_key}:loose:{data_hash}"
        
        return base_key
    
    def _normalize_time_range(self, cooking_time: Optional[int]) -> str:
        """Normalize cooking time into ranges for similar matching"""
        if not cooking_time:
            return "any"
        
        if cooking_time <= 15:
            return "quick"
        elif cooking_time <= 45:
            return "medium"
        else:
            return "long"
    
    async def intelligent_cache_lookup(
        self,
        request_data: Dict[str, Any],
        request_type: str
    ) -> Optional[Dict[str, Any]]:
        """Try multiple cache strategies to find a match"""
        
        base_key = f"chefood:{request_type}"
        
        # Try exact match first
        exact_key = await self.get_cache_key_with_strategy(
            base_key, request_data, "exact"
        )
        result = await self._get_from_cache(exact_key)
        
        if result:
            await self._track_cache_hit("exact")
            return result
        
        # Try similar match
        similar_key = await self.get_cache_key_with_strategy(
            base_key, request_data, "similar"
        )
        result = await self._get_from_cache(similar_key)
        
        if result:
            await self._track_cache_hit("similar")
            # Adapt result to current request
            return await self._adapt_cached_result(result, request_data)
        
        # Try loose match for free tier users
        loose_key = await self.get_cache_key_with_strategy(
            base_key, request_data, "loose"
        )
        result = await self._get_from_cache(loose_key)
        
        if result:
            await self._track_cache_hit("loose")
            return await self._adapt_cached_result(result, request_data)
        
        await self._track_cache_miss()
        return None
    
    async def _adapt_cached_result(
        self,
        cached_result: Dict[str, Any],
        current_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt cached result to match current request parameters"""
        
        # Simple adaptations
        if "servings" in current_request:
            cached_result["servings"] = current_request["servings"]
        
        if "cooking_time_minutes" in current_request:
            cached_result["total_time_minutes"] = current_request["cooking_time_minutes"]
        
        # Mark as adapted
        cached_result["_adapted"] = True
        cached_result["_original_cached"] = True
        
        return cached_result
    
    async def implement_request_coalescing(
        self,
        request_hash: str,
        timeout_seconds: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Coalesce identical requests to reduce API calls"""
        
        # Check if identical request is already in progress
        in_progress_key = f"in_progress:{request_hash}"
        
        # Try to set lock with expiration
        lock_acquired = await self.redis_client.set(
            in_progress_key,
            "processing",
            ex=timeout_seconds,
            nx=True
        )
        
        if not lock_acquired:
            # Another identical request is in progress, wait for result
            for _ in range(timeout_seconds * 2):
                result = await self.redis_client.get(f"result:{request_hash}")
                if result:
                    await self.redis_client.delete(f"result:{request_hash}")
                    return json.loads(result)
                await asyncio.sleep(0.5)
        
        return None  # No coalescing, proceed with request
    
    async def store_coalesced_result(
        self,
        request_hash: str,
        result: Dict[str, Any]
    ):
        """Store result for coalesced requests"""
        
        # Store result temporarily for other waiting requests
        await self.redis_client.setex(
            f"result:{request_hash}",
            30,  # 30 seconds
            json.dumps(result)
        )
        
        # Remove processing lock
        await self.redis_client.delete(f"in_progress:{request_hash}")
    
    async def track_usage_and_cost(
        self,
        user_id: str,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool = False
    ):
        """Track usage and calculate cost"""
        
        cost = 0.0
        if not cache_hit:
            pricing = self.pricing.get(model_used, self.pricing["gemini-1.5-flash"])
            cost = (
                (input_tokens / 1000) * pricing["input"] +
                (output_tokens / 1000) * pricing["output"]
            )
        
        # Update user metrics
        today = datetime.utcnow().date().isoformat()
        usage_key = f"usage:{user_id}:{today}"
        
        usage_data = {
            "requests_count": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cache_hits": 1 if cache_hit else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in Redis with daily aggregation
        await self.redis_client.hincrby(usage_key, "requests_count", 1)
        await self.redis_client.hincrbyfloat(usage_key, "cost", cost)
        await self.redis_client.hincrby(usage_key, "input_tokens", input_tokens)
        await self.redis_client.hincrby(usage_key, "output_tokens", output_tokens)
        
        if cache_hit:
            await self.redis_client.hincrby(usage_key, "cache_hits", 1)
        
        # Set expiration for 30 days
        await self.redis_client.expire(usage_key, 30 * 24 * 3600)
    
    async def get_user_usage_today(self, user_id: str) -> Dict[str, Any]:
        """Get user's usage for today"""
        
        today = datetime.utcnow().date().isoformat()
        usage_key = f"usage:{user_id}:{today}"
        
        usage_data = await self.redis_client.hgetall(usage_key)
        
        if not usage_data:
            return {
                "requests_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "cache_hits": 0
            }
        
        return {
            "requests_count": int(usage_data.get(b"requests_count", 0)),
            "input_tokens": int(usage_data.get(b"input_tokens", 0)),
            "output_tokens": int(usage_data.get(b"output_tokens", 0)),
            "cost": float(usage_data.get(b"cost", 0.0)),
            "cache_hits": int(usage_data.get(b"cache_hits", 0))
        }
    
    async def implement_smart_preloading(
        self,
        user_id: str,
        usage_patterns: Dict[str, Any]
    ):
        """Preload popular recipes based on usage patterns"""
        
        # Analyze user's common ingredients and preferences
        common_ingredients = usage_patterns.get("frequent_ingredients", [])
        common_dietary = usage_patterns.get("dietary_preferences", [])
        
        # Generate cache keys for likely requests
        popular_combinations = [
            {"ingredients": common_ingredients[:3], "dietary": common_dietary},
            {"ingredients": common_ingredients[:5], "dietary": common_dietary},
        ]
        
        # Preload in background during low-traffic hours
        for combo in popular_combinations:
            cache_key = await self.get_cache_key_with_strategy(
                "chefood:recipe", combo, "similar"
            )
            
            # Check if already cached
            if not await self._get_from_cache(cache_key):
                # Add to preload queue
                await self.redis_client.sadd("preload_queue", cache_key)
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
        return None
    
    async def _track_cache_hit(self, strategy: str):
        """Track cache hit metrics"""
        await self.redis_client.hincrby("cache_metrics", f"hits_{strategy}", 1)
    
    async def _track_cache_miss(self):
        """Track cache miss metrics"""
        await self.redis_client.hincrby("cache_metrics", "misses", 1)
    
    async def get_cost_optimization_report(self, user_id: str) -> Dict[str, Any]:
        """Generate cost optimization report for user"""
        
        # Get last 7 days of usage
        reports = []
        total_savings = 0.0
        
        for i in range(7):
            date = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
            usage = await self.get_user_usage_today(user_id)
            
            # Calculate potential savings from caching
            if usage["cache_hits"] > 0:
                avg_request_cost = usage["cost"] / max(usage["requests_count"], 1)
                savings = usage["cache_hits"] * avg_request_cost
                total_savings += savings
            
            reports.append({
                "date": date,
                "usage": usage,
                "savings": savings if usage["cache_hits"] > 0 else 0
            })
        
        # Get cache performance metrics
        cache_metrics = await self.redis_client.hgetall("cache_metrics")
        
        return {
            "period": "last_7_days",
            "daily_reports": reports,
            "total_savings": round(total_savings, 4),
            "cache_performance": {
                "hit_rate": self._calculate_cache_hit_rate(cache_metrics),
                "strategy_breakdown": {
                    "exact_hits": int(cache_metrics.get(b"hits_exact", 0)),
                    "similar_hits": int(cache_metrics.get(b"hits_similar", 0)),
                    "loose_hits": int(cache_metrics.get(b"hits_loose", 0)),
                    "misses": int(cache_metrics.get(b"misses", 0))
                }
            },
            "recommendations": await self._generate_cost_recommendations(user_id)
        }
    
    def _calculate_cache_hit_rate(self, metrics: Dict) -> float:
        """Calculate overall cache hit rate"""
        total_hits = (
            int(metrics.get(b"hits_exact", 0)) +
            int(metrics.get(b"hits_similar", 0)) +
            int(metrics.get(b"hits_loose", 0))
        )
        total_misses = int(metrics.get(b"misses", 0))
        total_requests = total_hits + total_misses
        
        if total_requests == 0:
            return 0.0
        
        return round((total_hits / total_requests) * 100, 2)
    
    async def _generate_cost_recommendations(self, user_id: str) -> List[str]:
        """Generate personalized cost optimization recommendations"""
        
        usage = await self.get_user_usage_today(user_id)
        recommendations = []
        
        if usage["cache_hits"] / max(usage["requests_count"], 1) < 0.3:
            recommendations.append(
                "Consider using more common ingredient combinations to benefit from caching"
            )
        
        if usage["cost"] > 5.0:  # High daily cost
            recommendations.append(
                "Try batch processing multiple recipes together to reduce costs"
            )
        
        if usage["requests_count"] > 50:
            recommendations.append(
                "Consider upgrading to a premium plan for better per-request pricing"
            )
        
        return recommendations
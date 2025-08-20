"""
A/B Testing and Prompt Optimization System for ChefoodAI
Enables systematic testing of different prompts, models, and parameters
"""

import json
import asyncio
import hashlib
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import logging
import uuid

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class MetricType(Enum):
    """Types of metrics to track"""
    RESPONSE_TIME = "response_time"
    USER_RATING = "user_rating"
    COMPLETION_RATE = "completion_rate"
    TOKEN_EFFICIENCY = "token_efficiency"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    USER_ENGAGEMENT = "user_engagement"
    RECIPE_COMPLEXITY = "recipe_complexity"

@dataclass
class ExperimentVariant:
    """A single variant in an A/B test"""
    id: str
    name: str
    description: str
    prompt_template: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    traffic_allocation: float = 0.0  # Percentage of traffic (0.0 to 1.0)
    is_control: bool = False

@dataclass
class Experiment:
    """A/B test experiment configuration"""
    id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: List[ExperimentVariant]
    target_metrics: List[MetricType]
    start_date: datetime
    end_date: Optional[datetime]
    minimum_sample_size: int = 100
    confidence_level: float = 0.95
    created_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    """Result of a single experiment execution"""
    experiment_id: str
    variant_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    metrics: Dict[str, float]
    user_feedback: Optional[Dict[str, Any]] = None

class ABTestManager:
    """Main A/B testing and experiment management system"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.active_experiments = {}
        self.experiment_results = defaultdict(list)
        self.prompt_library = PromptLibrary()
        
    async def create_experiment(self, experiment: Experiment) -> str:
        """Create a new A/B test experiment"""
        
        # Validate experiment configuration
        self._validate_experiment(experiment)
        
        # Store experiment configuration
        await self.redis_client.hset(
            f"experiment:{experiment.id}",
            mapping={
                "config": json.dumps(experiment.__dict__, default=str),
                "status": experiment.status.value,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # If active, add to active experiments cache
        if experiment.status == ExperimentStatus.ACTIVE:
            self.active_experiments[experiment.id] = experiment
        
        logger.info(f"Created experiment: {experiment.name} ({experiment.id})")
        return experiment.id
    
    def _validate_experiment(self, experiment: Experiment):
        """Validate experiment configuration"""
        
        # Check traffic allocation sums to 1.0
        total_allocation = sum(v.traffic_allocation for v in experiment.variants)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Ensure at least one control variant
        if not any(v.is_control for v in experiment.variants):
            raise ValueError("At least one variant must be marked as control")
        
        # Validate variant IDs are unique
        variant_ids = [v.id for v in experiment.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Variant IDs must be unique")
    
    async def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        request_context: Dict[str, Any] = None
    ) -> Optional[ExperimentVariant]:
        """Assign user to experiment variant"""
        
        experiment = await self.get_experiment(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.ACTIVE:
            return None
        
        # Check if user already assigned
        existing_assignment = await self.redis_client.get(
            f"assignment:{experiment_id}:{user_id}"
        )
        
        if existing_assignment:
            variant_id = existing_assignment.decode()
            return next(
                (v for v in experiment.variants if v.id == variant_id),
                None
            )
        
        # Assign new variant using consistent hashing
        variant = self._assign_variant_deterministic(
            experiment,
            user_id,
            request_context
        )
        
        # Store assignment
        await self.redis_client.setex(
            f"assignment:{experiment_id}:{user_id}",
            30 * 24 * 3600,  # 30 days
            variant.id
        )
        
        # Track assignment
        await self._track_assignment(experiment_id, variant.id, user_id)
        
        return variant
    
    def _assign_variant_deterministic(
        self,
        experiment: Experiment,
        user_id: str,
        request_context: Dict[str, Any] = None
    ) -> ExperimentVariant:
        """Deterministically assign variant based on user ID"""
        
        # Create hash for consistent assignment
        hash_input = f"{experiment.id}:{user_id}"
        if request_context and experiment.metadata.get("segment_by_request"):
            # Optionally segment by request characteristics
            segment_keys = experiment.metadata.get("segment_keys", [])
            segment_data = {k: request_context.get(k) for k in segment_keys}
            hash_input += f":{json.dumps(segment_data, sort_keys=True)}"
        
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        random.seed(hash_value)
        
        # Select variant based on traffic allocation
        rand_value = random.random()
        cumulative_allocation = 0.0
        
        for variant in experiment.variants:
            cumulative_allocation += variant.traffic_allocation
            if rand_value <= cumulative_allocation:
                return variant
        
        # Fallback to control variant
        return next(v for v in experiment.variants if v.is_control)
    
    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment configuration"""
        
        # Check cache first
        if experiment_id in self.active_experiments:
            return self.active_experiments[experiment_id]
        
        # Load from Redis
        experiment_data = await self.redis_client.hget(
            f"experiment:{experiment_id}",
            "config"
        )
        
        if not experiment_data:
            return None
        
        config = json.loads(experiment_data)
        
        # Reconstruct experiment object
        experiment = Experiment(
            id=config["id"],
            name=config["name"],
            description=config["description"],
            status=ExperimentStatus(config["status"]),
            variants=[
                ExperimentVariant(**v) for v in config["variants"]
            ],
            target_metrics=[MetricType(m) for m in config["target_metrics"]],
            start_date=datetime.fromisoformat(config["start_date"]),
            end_date=datetime.fromisoformat(config["end_date"]) if config.get("end_date") else None,
            minimum_sample_size=config.get("minimum_sample_size", 100),
            confidence_level=config.get("confidence_level", 0.95),
            created_by=config.get("created_by", ""),
            metadata=config.get("metadata", {})
        )
        
        return experiment
    
    async def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        metrics: Dict[str, float],
        user_feedback: Dict[str, Any] = None
    ):
        """Record experiment result"""
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            request_data=request_data,
            response_data=response_data,
            metrics=metrics,
            user_feedback=user_feedback
        )
        
        # Store result
        result_key = f"results:{experiment_id}:{variant_id}"
        await self.redis_client.lpush(
            result_key,
            json.dumps(result.__dict__, default=str)
        )
        
        # Set expiration (90 days)
        await self.redis_client.expire(result_key, 90 * 24 * 3600)
        
        # Update metrics aggregation
        await self._update_metrics_aggregation(experiment_id, variant_id, metrics)
        
        logger.debug(f"Recorded result for experiment {experiment_id}, variant {variant_id}")
    
    async def _update_metrics_aggregation(
        self,
        experiment_id: str,
        variant_id: str,
        metrics: Dict[str, float]
    ):
        """Update aggregated metrics for real-time analysis"""
        
        for metric_name, value in metrics.items():
            key = f"metrics:{experiment_id}:{variant_id}:{metric_name}"
            
            # Store individual values for statistical analysis
            await self.redis_client.lpush(key, str(value))
            await self.redis_client.expire(key, 90 * 24 * 3600)
            
            # Update aggregated statistics
            await self._update_metric_stats(key, value)
    
    async def _update_metric_stats(self, base_key: str, value: float):
        """Update running statistics for a metric"""
        
        stats_key = f"{base_key}:stats"
        
        # Get current stats
        current_stats = await self.redis_client.hmget(
            stats_key,
            "count", "sum", "sum_squares", "min", "max"
        )
        
        count = int(current_stats[0] or 0) + 1
        total = float(current_stats[1] or 0) + value
        sum_squares = float(current_stats[2] or 0) + (value * value)
        min_val = min(float(current_stats[3] or float('inf')), value)
        max_val = max(float(current_stats[4] or float('-inf')), value)
        
        # Update stats
        await self.redis_client.hmset(
            stats_key,
            {
                "count": count,
                "sum": total,
                "sum_squares": sum_squares,
                "min": min_val,
                "max": max_val,
                "mean": total / count,
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        await self.redis_client.expire(stats_key, 90 * 24 * 3600)
    
    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results and determine statistical significance"""
        
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        analysis = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "analysis_date": datetime.utcnow().isoformat(),
            "variants": {},
            "statistical_significance": {},
            "recommendations": []
        }
        
        # Analyze each variant
        for variant in experiment.variants:
            variant_analysis = await self._analyze_variant(
                experiment_id,
                variant,
                experiment.target_metrics
            )
            analysis["variants"][variant.id] = variant_analysis
        
        # Perform statistical significance tests
        control_variant = next(v for v in experiment.variants if v.is_control)
        
        for variant in experiment.variants:
            if variant.is_control:
                continue
            
            significance_results = await self._test_statistical_significance(
                experiment_id,
                control_variant,
                variant,
                experiment.target_metrics,
                experiment.confidence_level
            )
            
            analysis["statistical_significance"][variant.id] = significance_results
        
        # Generate recommendations
        analysis["recommendations"] = await self._generate_recommendations(
            experiment,
            analysis
        )
        
        return analysis
    
    async def _analyze_variant(
        self,
        experiment_id: str,
        variant: ExperimentVariant,
        target_metrics: List[MetricType]
    ) -> Dict[str, Any]:
        """Analyze performance of a single variant"""
        
        variant_analysis = {
            "variant_id": variant.id,
            "variant_name": variant.name,
            "is_control": variant.is_control,
            "traffic_allocation": variant.traffic_allocation,
            "metrics": {}
        }
        
        for metric_type in target_metrics:
            metric_name = metric_type.value
            stats_key = f"metrics:{experiment_id}:{variant.id}:{metric_name}:stats"
            
            stats = await self.redis_client.hmget(
                stats_key,
                "count", "mean", "min", "max", "sum", "sum_squares"
            )
            
            if stats[0]:  # Has data
                count = int(stats[0])
                mean = float(stats[1])
                min_val = float(stats[2])
                max_val = float(stats[3])
                total = float(stats[4])
                sum_squares = float(stats[5])
                
                # Calculate standard deviation
                variance = (sum_squares - (total * total) / count) / (count - 1) if count > 1 else 0
                std_dev = variance ** 0.5 if variance >= 0 else 0
                
                variant_analysis["metrics"][metric_name] = {
                    "count": count,
                    "mean": round(mean, 4),
                    "std_dev": round(std_dev, 4),
                    "min": round(min_val, 4),
                    "max": round(max_val, 4),
                    "confidence_interval": self._calculate_confidence_interval(
                        mean, std_dev, count, 0.95
                    )
                }
            else:
                variant_analysis["metrics"][metric_name] = {
                    "count": 0,
                    "mean": 0,
                    "message": "No data available"
                }
        
        return variant_analysis
    
    def _calculate_confidence_interval(
        self,
        mean: float,
        std_dev: float,
        count: int,
        confidence: float
    ) -> List[float]:
        """Calculate confidence interval for metric"""
        
        if count <= 1:
            return [mean, mean]
        
        # Use t-distribution for small samples, normal for large
        if count < 30:
            # Simplified t-value (should use actual t-table)
            t_value = 2.0  # Approximate for 95% confidence
        else:
            t_value = 1.96  # Z-value for 95% confidence
        
        margin_error = t_value * (std_dev / (count ** 0.5))
        
        return [
            round(mean - margin_error, 4),
            round(mean + margin_error, 4)
        ]
    
    async def _test_statistical_significance(
        self,
        experiment_id: str,
        control_variant: ExperimentVariant,
        test_variant: ExperimentVariant,
        target_metrics: List[MetricType],
        confidence_level: float
    ) -> Dict[str, Any]:
        """Test statistical significance between control and test variants"""
        
        significance_results = {}
        
        for metric_type in target_metrics:
            metric_name = metric_type.value
            
            # Get raw data for both variants
            control_data = await self._get_metric_data(
                experiment_id,
                control_variant.id,
                metric_name
            )
            test_data = await self._get_metric_data(
                experiment_id,
                test_variant.id,
                metric_name
            )
            
            if len(control_data) < 10 or len(test_data) < 10:
                significance_results[metric_name] = {
                    "significant": False,
                    "p_value": None,
                    "message": "Insufficient data for significance testing"
                }
                continue
            
            # Perform t-test (simplified)
            p_value = self._welch_t_test(control_data, test_data)
            is_significant = p_value < (1 - confidence_level)
            
            # Calculate effect size
            effect_size = self._calculate_effect_size(control_data, test_data)
            
            # Calculate relative improvement
            control_mean = statistics.mean(control_data)
            test_mean = statistics.mean(test_data)
            relative_improvement = ((test_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0
            
            significance_results[metric_name] = {
                "significant": is_significant,
                "p_value": round(p_value, 6),
                "effect_size": round(effect_size, 4),
                "relative_improvement_percent": round(relative_improvement, 2),
                "control_mean": round(control_mean, 4),
                "test_mean": round(test_mean, 4),
                "sample_sizes": {
                    "control": len(control_data),
                    "test": len(test_data)
                }
            }
        
        return significance_results
    
    async def _get_metric_data(
        self,
        experiment_id: str,
        variant_id: str,
        metric_name: str
    ) -> List[float]:
        """Get raw metric data for statistical analysis"""
        
        key = f"metrics:{experiment_id}:{variant_id}:{metric_name}"
        raw_data = await self.redis_client.lrange(key, 0, -1)
        
        return [float(value) for value in raw_data if value]
    
    def _welch_t_test(self, sample1: List[float], sample2: List[float]) -> float:
        """Simplified Welch's t-test implementation"""
        
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        var1 = statistics.variance(sample1) if n1 > 1 else 0
        var2 = statistics.variance(sample2) if n2 > 1 else 0
        
        if var1 == 0 and var2 == 0:
            return 1.0 if mean1 == mean2 else 0.0
        
        # Welch's t-statistic
        pooled_se = ((var1 / n1) + (var2 / n2)) ** 0.5
        if pooled_se == 0:
            return 1.0 if mean1 == mean2 else 0.0
        
        t_stat = abs((mean1 - mean2) / pooled_se)
        
        # Simplified p-value calculation (should use proper t-distribution)
        # This is a rough approximation
        if t_stat > 2.0:
            return 0.02  # Significant
        elif t_stat > 1.5:
            return 0.1   # Moderate
        else:
            return 0.5   # Not significant
    
    def _calculate_effect_size(self, sample1: List[float], sample2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        var1 = statistics.variance(sample1) if len(sample1) > 1 else 0
        var2 = statistics.variance(sample2) if len(sample2) > 1 else 0
        
        # Pooled standard deviation
        pooled_std = ((var1 + var2) / 2) ** 0.5
        
        if pooled_std == 0:
            return 0.0
        
        return (mean2 - mean1) / pooled_std
    
    async def _generate_recommendations(
        self,
        experiment: Experiment,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Check sample sizes
        min_sample_size = experiment.minimum_sample_size
        for variant_id, variant_data in analysis["variants"].items():
            for metric_name, metric_data in variant_data["metrics"].items():
                if metric_data.get("count", 0) < min_sample_size:
                    recommendations.append(
                        f"Collect more data for {variant_data['variant_name']} "
                        f"({metric_data.get('count', 0)}/{min_sample_size} samples)"
                    )
        
        # Check for significant improvements
        significant_improvements = []
        for variant_id, significance_data in analysis["statistical_significance"].items():
            variant_name = next(
                v.name for v in experiment.variants if v.id == variant_id
            )
            
            for metric_name, metric_significance in significance_data.items():
                if metric_significance.get("significant") and metric_significance.get("relative_improvement_percent", 0) > 0:
                    significant_improvements.append(
                        f"{variant_name} shows {metric_significance['relative_improvement_percent']:.1f}% "
                        f"improvement in {metric_name}"
                    )
        
        if significant_improvements:
            recommendations.append("Consider implementing winning variants:")
            recommendations.extend(significant_improvements)
        
        # Check experiment duration
        if experiment.end_date and datetime.utcnow() > experiment.end_date:
            recommendations.append("Experiment has reached end date - consider concluding")
        
        return recommendations
    
    async def _track_assignment(self, experiment_id: str, variant_id: str, user_id: str):
        """Track variant assignment for analytics"""
        
        assignment_key = f"assignments:{experiment_id}:{variant_id}"
        await self.redis_client.sadd(assignment_key, user_id)
        await self.redis_client.expire(assignment_key, 90 * 24 * 3600)
    
    async def get_experiment_dashboard(self, experiment_id: str) -> Dict[str, Any]:
        """Get real-time experiment dashboard data"""
        
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        dashboard = {
            "experiment": {
                "id": experiment.id,
                "name": experiment.name,
                "status": experiment.status.value,
                "start_date": experiment.start_date.isoformat(),
                "end_date": experiment.end_date.isoformat() if experiment.end_date else None
            },
            "real_time_metrics": {},
            "traffic_distribution": {},
            "recent_activity": []
        }
        
        # Get real-time metrics for each variant
        for variant in experiment.variants:
            variant_metrics = {}
            
            for metric_type in experiment.target_metrics:
                metric_name = metric_type.value
                stats = await self._get_real_time_metric_stats(
                    experiment_id,
                    variant.id,
                    metric_name
                )
                variant_metrics[metric_name] = stats
            
            dashboard["real_time_metrics"][variant.id] = {
                "variant_name": variant.name,
                "metrics": variant_metrics
            }
            
            # Get traffic count
            assignment_count = await self.redis_client.scard(
                f"assignments:{experiment_id}:{variant.id}"
            )
            dashboard["traffic_distribution"][variant.id] = {
                "variant_name": variant.name,
                "assignment_count": assignment_count,
                "expected_allocation": variant.traffic_allocation
            }
        
        return dashboard
    
    async def _get_real_time_metric_stats(
        self,
        experiment_id: str,
        variant_id: str,
        metric_name: str
    ) -> Dict[str, Any]:
        """Get real-time metric statistics"""
        
        stats_key = f"metrics:{experiment_id}:{variant_id}:{metric_name}:stats"
        stats = await self.redis_client.hmget(
            stats_key,
            "count", "mean", "min", "max"
        )
        
        if stats[0]:
            return {
                "count": int(stats[0]),
                "mean": round(float(stats[1]), 4),
                "min": round(float(stats[2]), 4),
                "max": round(float(stats[3]), 4)
            }
        else:
            return {"count": 0, "mean": 0, "min": 0, "max": 0}

class PromptLibrary:
    """Library of optimized prompts for different use cases"""
    
    def __init__(self):
        self.templates = self._load_prompt_templates()
        self.optimization_strategies = self._load_optimization_strategies()
    
    def get_prompt_variants(self, base_prompt_type: str) -> List[Dict[str, Any]]:
        """Get different variants of a prompt for A/B testing"""
        
        base_template = self.templates.get(base_prompt_type, {})
        variants = []
        
        # Create variants using different optimization strategies
        for strategy_name, strategy in self.optimization_strategies.items():
            variant_prompt = self._apply_strategy(base_template, strategy)
            
            variants.append({
                "id": f"{base_prompt_type}_{strategy_name}",
                "name": f"{base_prompt_type.title()} - {strategy_name.title()}",
                "prompt_template": variant_prompt,
                "strategy": strategy_name,
                "expected_improvement": strategy.get("expected_improvement", "unknown")
            })
        
        return variants
    
    def _apply_strategy(
        self,
        base_template: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> str:
        """Apply optimization strategy to base prompt"""
        
        prompt = base_template.get("template", "")
        
        # Apply strategy modifications
        if strategy.get("add_context"):
            prompt = strategy["context_prefix"] + "\n\n" + prompt
        
        if strategy.get("modify_tone"):
            prompt = prompt.replace(
                base_template.get("tone_marker", ""),
                strategy["tone_replacement"]
            )
        
        if strategy.get("add_examples"):
            prompt += "\n\n" + strategy["examples_section"]
        
        if strategy.get("modify_output_format"):
            prompt = prompt.replace(
                base_template.get("format_section", ""),
                strategy["format_replacement"]
            )
        
        return prompt
    
    def _load_prompt_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load base prompt templates"""
        
        return {
            "recipe_generation": {
                "template": """You are ChefoodAI, an expert culinary assistant. Generate a detailed recipe based on the following requirements:

**Available Ingredients:** {ingredients}
**Dietary Preferences:** {dietary_preferences}
**Cuisine Type:** {cuisine_type}
**Cooking Time:** {cooking_time} minutes
**Skill Level:** {skill_level}
**Servings:** {servings}

Please provide a complete recipe with ingredients, instructions, and nutritional information.""",
                "tone_marker": "You are ChefoodAI, an expert culinary assistant.",
                "format_section": "Please provide a complete recipe"
            },
            
            "meal_planning": {
                "template": """Create a comprehensive meal plan for {days} days with the following requirements:

**Dietary Restrictions:** {dietary_preferences}
**Family Size:** {family_size}
**Budget:** {budget}
**Prep Time Preference:** {prep_time}

Generate a balanced meal plan with variety and nutritional balance.""",
                "tone_marker": "Create a comprehensive meal plan",
                "format_section": "Generate a balanced meal plan"
            }
        }
    
    def _load_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load prompt optimization strategies"""
        
        return {
            "detailed_context": {
                "add_context": True,
                "context_prefix": "Consider seasonal ingredients, cooking techniques, and flavor profiles when creating recipes. Focus on practical, achievable results.",
                "expected_improvement": "Better recipe quality and practicality"
            },
            
            "conversational": {
                "modify_tone": True,
                "tone_replacement": "Hi! I'm ChefoodAI, your friendly kitchen companion. Let me help you create something delicious!",
                "expected_improvement": "Higher user engagement"
            },
            
            "structured_output": {
                "modify_output_format": True,
                "format_replacement": """Return the recipe in this exact JSON format:
{
  "title": "Recipe Name",
  "description": "Brief description",
  "ingredients": [...],
  "instructions": [...],
  "nutrition": {...}
}""",
                "expected_improvement": "More consistent response format"
            },
            
            "example_driven": {
                "add_examples": True,
                "examples_section": """Example of a great recipe format:
Title: "Creamy Garlic Pasta"
Description: "A rich, comforting pasta dish with aromatic garlic"
Prep Time: 10 minutes
Cook Time: 15 minutes""",
                "expected_improvement": "Better adherence to desired format"
            },
            
            "efficiency_focused": {
                "modify_tone": True,
                "tone_replacement": "Generate an efficient, practical recipe focusing on:",
                "add_context": True,
                "context_prefix": "Prioritize: 1) Minimal prep time 2) Common ingredients 3) Simple techniques 4) Clear instructions",
                "expected_improvement": "Faster, more practical recipes"
            }
        }
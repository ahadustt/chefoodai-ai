"""
Configuration management for ChefoodAI AI Service
Centralized settings with proper validation and security
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, validator, Field
from enum import Enum


class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Application
    app_name: str = "ChefoodAI AI Service"
    app_version: str = "2.0.0"
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=10)
    
    # Security
    cors_origins: List[str] = Field(default_factory=lambda: ["https://chefoodai.com", "https://app.chefoodai.com"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_allow_headers: List[str] = Field(default_factory=lambda: ["*"])
    allowed_hosts: List[str] = Field(default_factory=lambda: ["*"])
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=60, ge=1)  # seconds
    max_request_size: int = Field(default=10 * 1024 * 1024, ge=1024)  # 10MB
    
    # Google Cloud
    google_cloud_project: str = Field(..., min_length=1)
    google_cloud_region: str = "us-central1"
    vertex_ai_location: str = "us-central1"
    
    # Redis Configuration
    redis_url: Optional[str] = None
    redis_max_connections: int = Field(default=20, ge=1)
    redis_retry_on_timeout: bool = True
    redis_socket_timeout: int = Field(default=5, ge=1)
    
    # AI Service Configuration
    default_model: str = "gemini-1.5-flash"
    fallback_model: str = "gemini-1.0-pro"
    max_tokens: int = Field(default=8192, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # Cache Configuration
    cache_ttl_default: int = Field(default=3600, ge=60)  # 1 hour
    cache_ttl_recipes: int = Field(default=7200, ge=60)  # 2 hours
    cache_ttl_images: int = Field(default=86400, ge=60)  # 24 hours
    
    # Monitoring & Logging
    log_level: LogLevel = LogLevel.INFO
    enable_access_logs: bool = True
    enable_metrics: bool = True
    metrics_port: int = Field(default=9090, ge=1, le=65535)
    
    # API Keys (loaded from environment or secrets)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Timeouts
    http_timeout: int = Field(default=30, ge=1)
    ai_request_timeout: int = Field(default=120, ge=10)
    
    # Feature Flags
    enable_cost_optimization: bool = True
    enable_fallback_strategies: bool = True
    enable_ab_testing: bool = False
    enable_image_generation: bool = True
    enable_advanced_nutrition: bool = True
    
    # Performance
    max_concurrent_requests: int = Field(default=100, ge=1)
    request_timeout: int = Field(default=300, ge=30)  # 5 minutes
    keep_alive_timeout: int = Field(default=5, ge=1)
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        if v == Environment.PRODUCTION:
            # Additional validation for production
            pass
        return v
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v, values):
        env = values.get('environment')
        if env == Environment.PRODUCTION and '*' in v:
            raise ValueError("Wildcard CORS origins not allowed in production")
        return v
    
    @validator('debug')
    def validate_debug(cls, v, values):
        env = values.get('environment')
        if env == Environment.PRODUCTION and v:
            raise ValueError("Debug mode not allowed in production")
        return v
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        return self.environment == Environment.TESTING
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Map environment variables
        fields = {
            "google_cloud_project": {"env": "GOOGLE_CLOUD_PROJECT"},
            "redis_url": {"env": "REDIS_URL"},
            "openai_api_key": {"env": "OPENAI_API_KEY"},
            "anthropic_api_key": {"env": "ANTHROPIC_API_KEY"},
            "cors_origins": {"env": "CORS_ORIGINS"},
            "port": {"env": "PORT"},
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


# Validation functions
def validate_production_config(config: Settings) -> List[str]:
    """Validate configuration for production deployment"""
    issues = []
    
    if config.debug:
        issues.append("Debug mode should be disabled in production")
    
    if "*" in config.cors_origins:
        issues.append("CORS origins should not include wildcards in production")
    
    if not config.google_cloud_project:
        issues.append("Google Cloud Project must be specified")
    
    if config.log_level == LogLevel.DEBUG:
        issues.append("Log level should not be DEBUG in production")
    
    return issues
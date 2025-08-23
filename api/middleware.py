"""
Middleware for ChefoodAI AI Service
Comprehensive security, monitoring, and performance middleware
"""

import time
import uuid
import json
import asyncio
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import structlog

from config import get_settings
from .models import ErrorResponse, ErrorType, ResponseStatus

logger = structlog.get_logger()
settings = get_settings()


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Add request tracing and timing information"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to headers
        request.state.request_id = request_id
        
        # Log request start
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            response.headers["X-Service"] = "chefoodai-ai-service"
            
            # Log request completion
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                process_time=round(process_time, 4),
            )
            
            return response
            
        except Exception as exc:
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(exc),
                error_type=type(exc).__name__,
                process_time=round(process_time, 4),
            )
            
            # Create standardized error response
            error_response = ErrorResponse(
                error_type=ErrorType.INTERNAL_ERROR,
                detail=str(exc),
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id}
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with sliding window"""
    
    def __init__(self, app, requests_per_minute: int = None, requests_per_hour: int = None):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or settings.rate_limit_requests
        self.requests_per_hour = requests_per_hour or (settings.rate_limit_requests * 60)
        
        # Store request timestamps per IP
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Cleanup interval
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def _cleanup_old_requests(self):
        """Remove old request records"""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return
            
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour
        
        for ip in list(self.request_history.keys()):
            history = self.request_history[ip]
            # Remove requests older than 1 hour
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # Remove empty histories
            if not history:
                del self.request_history[ip]
        
        self.last_cleanup = current_time
    
    def _is_rate_limited(self, client_ip: str) -> Optional[str]:
        """Check if client is rate limited"""
        current_time = time.time()
        history = self.request_history[client_ip]
        
        # Count requests in the last minute
        minute_ago = current_time - 60
        minute_requests = sum(1 for timestamp in history if timestamp > minute_ago)
        
        if minute_requests >= self.requests_per_minute:
            return f"Rate limit exceeded: {minute_requests}/{self.requests_per_minute} requests per minute"
        
        # Count requests in the last hour
        hour_ago = current_time - 3600
        hour_requests = sum(1 for timestamp in history if timestamp > hour_ago)
        
        if hour_requests >= self.requests_per_hour:
            return f"Rate limit exceeded: {hour_requests}/{self.requests_per_hour} requests per hour"
        
        return None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean up old requests periodically
        self._cleanup_old_requests()
        
        # Check rate limit
        rate_limit_message = self._is_rate_limited(client_ip)
        if rate_limit_message:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                message=rate_limit_message,
                request_id=getattr(request.state, 'request_id', None)
            )
            
            error_response = ErrorResponse(
                error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                detail=rate_limit_message,
                request_id=getattr(request.state, 'request_id', str(uuid.uuid4()))
            )
            
            return JSONResponse(
                status_code=429,
                content=error_response.dict(),
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + 60))
                }
            )
        
        # Record this request
        self.request_history[client_ip].append(time.time())
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Calculate remaining requests
        current_time = time.time()
        minute_ago = current_time - 60
        minute_requests = sum(1 for timestamp in self.request_history[client_ip] 
                            if timestamp > minute_ago)
        remaining = max(0, self.requests_per_minute - minute_requests)
        
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "microphone=(), camera=(), geolocation=()"
        
        # Only add HSTS in production with HTTPS
        if settings.is_production and request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy for API
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"
        
        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Limit request body size"""
    
    def __init__(self, app, max_size: int = None):
        super().__init__(app)
        self.max_size = max_size or settings.max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                content_length = int(content_length)
                if content_length > self.max_size:
                    error_response = ErrorResponse(
                        error_type=ErrorType.VALIDATION_ERROR,
                        detail=f"Request body too large. Maximum size: {self.max_size} bytes",
                        request_id=getattr(request.state, 'request_id', str(uuid.uuid4()))
                    )
                    
                    return JSONResponse(
                        status_code=413,
                        content=error_response.dict()
                    )
            except ValueError:
                pass
        
        return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect application metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_count = defaultdict(int)
        self.active_requests = 0
        
        # Metrics cleanup
        self.last_metrics_cleanup = time.time()
        self.metrics_cleanup_interval = 3600  # 1 hour
    
    def _cleanup_metrics(self):
        """Clean up old metrics data"""
        if time.time() - self.last_metrics_cleanup < self.metrics_cleanup_interval:
            return
        
        # Keep only recent response times (last 1000 per endpoint)
        for endpoint in self.response_times:
            if len(self.response_times[endpoint]) > 1000:
                self.response_times[endpoint] = self.response_times[endpoint][-1000:]
        
        self.last_metrics_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        self.active_requests += 1
        
        try:
            response = await call_next(request)
            
            # Record metrics
            self.request_count[endpoint] += 1
            response_time = time.time() - start_time
            self.response_times[endpoint].append(response_time)
            
            # Record errors
            if response.status_code >= 400:
                self.error_count[endpoint] += 1
            
            # Add metrics to response headers (optional, for debugging)
            if settings.environment != "production":
                response.headers["X-Metrics-Active-Requests"] = str(self.active_requests - 1)
                response.headers["X-Metrics-Total-Requests"] = str(sum(self.request_count.values()))
            
            return response
            
        except Exception as exc:
            # Record error
            self.error_count[endpoint] += 1
            raise
            
        finally:
            self.active_requests -= 1
            self._cleanup_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        total_requests = sum(self.request_count.values())
        total_errors = sum(self.error_count.values())
        
        # Calculate average response times
        avg_response_times = {}
        for endpoint, times in self.response_times.items():
            if times:
                avg_response_times[endpoint] = sum(times) / len(times)
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "active_requests": self.active_requests,
            "request_count_by_endpoint": dict(self.request_count),
            "error_count_by_endpoint": dict(self.error_count),
            "avg_response_times": avg_response_times,
            "timestamp": datetime.utcnow().isoformat()
        }


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Request timeout middleware"""
    
    def __init__(self, app, timeout: int = None):
        super().__init__(app)
        self.timeout = timeout or settings.request_timeout
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            # Use asyncio.wait_for to enforce timeout
            return await asyncio.wait_for(
                call_next(request), 
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Request timeout",
                timeout=self.timeout,
                request_id=getattr(request.state, 'request_id', None),
                path=request.url.path
            )
            
            error_response = ErrorResponse(
                error_type=ErrorType.TIMEOUT_ERROR,
                detail=f"Request timeout after {self.timeout} seconds",
                request_id=getattr(request.state, 'request_id', str(uuid.uuid4()))
            )
            
            return JSONResponse(
                status_code=504,
                content=error_response.dict()
            )


# Global metrics instance for access by other modules
metrics_middleware_instance: Optional[MetricsMiddleware] = None


def get_metrics() -> Dict[str, Any]:
    """Get current application metrics"""
    if metrics_middleware_instance:
        return metrics_middleware_instance.get_metrics()
    return {"error": "Metrics not available"}
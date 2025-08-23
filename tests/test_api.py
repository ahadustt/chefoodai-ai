"""
API endpoint tests
Comprehensive testing of all API routes and functionality
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestHealthEndpoints:
    """Test health and monitoring endpoints"""
    
    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint returns service information"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "capabilities" in data
        assert isinstance(data["capabilities"], list)
    
    def test_health_check(self, client: TestClient):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May be unhealthy without real services
        
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "checks" in data
        assert "uptime" in data
    
    def test_readiness_check(self, client: TestClient):
        """Test readiness probe endpoint"""
        response = client.get("/ready")
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
        assert "checks" in data
    
    def test_ping_endpoint(self, client: TestClient):
        """Test simple ping endpoint"""
        response = client.get("/ping")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "pong"
        assert "timestamp" in data
    
    def test_metrics_endpoint(self, client: TestClient):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code in [200, 404]  # May be disabled in test
        
        if response.status_code == 200:
            data = response.json()
            assert "service_info" in data


class TestRecipeEndpoints:
    """Test recipe generation endpoints"""
    
    def test_generate_recipe_success(self, client: TestClient, sample_recipe_request):
        """Test successful recipe generation"""
        response = client.post("/api/v1/recipes/generate", json=sample_recipe_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "recipe" in data
        assert "model_used" in data
        
        recipe = data["recipe"]
        assert "title" in recipe
        assert "ingredients" in recipe
        assert "instructions" in recipe
        assert "servings" in recipe
    
    def test_generate_recipe_validation_error(self, client: TestClient):
        """Test recipe generation with invalid input"""
        invalid_request = {
            "prompt": "ab",  # Too short
            "servings": 0    # Invalid
        }
        
        response = client.post("/api/v1/recipes/generate", json=invalid_request)
        assert response.status_code == 422
        
        data = response.json()
        assert "error_type" in data
        assert data["error_type"] == "validation_error"
    
    def test_get_cuisines(self, client: TestClient):
        """Test get supported cuisines"""
        response = client.get("/api/v1/recipes/cuisines")
        assert response.status_code == 200
        
        data = response.json()
        assert "cuisines" in data
        assert "total" in data
        assert isinstance(data["cuisines"], list)
    
    def test_get_dietary_restrictions(self, client: TestClient):
        """Test get dietary restrictions"""
        response = client.get("/api/v1/recipes/dietary-restrictions")
        assert response.status_code == 200
        
        data = response.json()
        assert "dietary_restrictions" in data
        assert "total" in data
    
    def test_get_difficulty_levels(self, client: TestClient):
        """Test get difficulty levels"""
        response = client.get("/api/v1/recipes/difficulty-levels")
        assert response.status_code == 200
        
        data = response.json()
        assert "difficulty_levels" in data
        assert "total" in data
    
    def test_enhance_recipe(self, client: TestClient):
        """Test recipe enhancement"""
        recipe_data = {
            "title": "Pasta Primavera",
            "ingredients": ["pasta", "vegetables", "olive oil"],
            "instructions": ["Cook pasta", "Sauté vegetables", "Combine"]
        }
        
        response = client.post("/api/v1/recipes/enhance", json=recipe_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "enhanced_recipe" in data


class TestMiddleware:
    """Test middleware functionality"""
    
    def test_request_id_header(self, client: TestClient):
        """Test that request ID is added to response headers"""
        response = client.get("/")
        assert "X-Request-ID" in response.headers
    
    def test_process_time_header(self, client: TestClient):
        """Test that process time is added to response headers"""
        response = client.get("/")
        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) >= 0
    
    def test_security_headers(self, client: TestClient):
        """Test security headers are added"""
        response = client.get("/")
        
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Referrer-Policy"
        ]
        
        for header in security_headers:
            assert header in response.headers
    
    def test_rate_limiting_headers(self, client: TestClient):
        """Test rate limiting headers are present"""
        response = client.get("/")
        
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ]
        
        for header in rate_limit_headers:
            assert header in response.headers
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are configured"""
        # Test preflight request
        response = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        })
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers


class TestErrorHandling:
    """Test error handling and validation"""
    
    def test_404_not_found(self, client: TestClient):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        data = response.json()
        assert "error_type" in data
        assert "detail" in data
        assert "request_id" in data
    
    def test_405_method_not_allowed(self, client: TestClient):
        """Test 405 error handling"""
        response = client.patch("/health")  # PATCH not allowed
        assert response.status_code == 405
    
    def test_422_validation_error(self, client: TestClient):
        """Test validation error handling"""
        invalid_recipe = {
            "prompt": "",  # Empty prompt
            "servings": -1  # Negative servings
        }
        
        response = client.post("/api/v1/recipes/generate", json=invalid_recipe)
        assert response.status_code == 422
        
        data = response.json()
        assert data["error_type"] == "validation_error"
        assert "errors" in data
        assert isinstance(data["errors"], list)
    
    def test_413_request_too_large(self, client: TestClient):
        """Test request size limit"""
        # Create a large request (> 10MB would be rejected by middleware)
        large_prompt = "a" * 1000000  # 1MB prompt
        
        large_request = {
            "prompt": large_prompt,
            "servings": 4
        }
        
        response = client.post("/api/v1/recipes/generate", json=large_request)
        # Should either process or return 413 depending on middleware configuration
        assert response.status_code in [200, 413, 422]


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async functionality"""
    
    async def test_async_recipe_generation(self, async_client: AsyncClient, sample_recipe_request):
        """Test async recipe generation"""
        response = await async_client.post("/api/v1/recipes/generate", json=sample_recipe_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "recipe" in data
    
    async def test_concurrent_requests(self, async_client: AsyncClient, sample_recipe_request):
        """Test handling of concurrent requests"""
        import asyncio
        
        # Send multiple concurrent requests
        tasks = [
            async_client.post("/api/v1/recipes/generate", json=sample_recipe_request)
            for _ in range(3)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "recipe" in data


class TestConfiguration:
    """Test configuration and settings"""
    
    def test_test_environment(self, test_settings):
        """Test that we're running in test environment"""
        assert test_settings.environment.value == "testing"
        assert test_settings.is_testing
        assert not test_settings.is_production
    
    def test_cors_configuration(self, test_settings):
        """Test CORS configuration for testing"""
        allowed_origins = test_settings.cors_origins
        assert "http://localhost:3000" in allowed_origins
        assert "http://localhost:8000" in allowed_origins
    
    def test_google_cloud_project_set(self, test_settings):
        """Test that Google Cloud project is set for testing"""
        assert test_settings.google_cloud_project == "test-project"
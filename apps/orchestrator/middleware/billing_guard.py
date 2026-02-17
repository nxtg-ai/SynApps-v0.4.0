"""
BillingGuard middleware for SynApps commercialization

This module implements middleware for handling rate-limiting and tier checks
for the freemium model.
"""
import os
import time
from typing import Dict, Optional, Callable
from fastapi import Request, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Constants for different tiers
FREE_TIER_MAX_RUNS = 20  # Maximum number of workflow runs per month for free tier
FREE_TIER_MAX_APPLETS = 3  # Maximum number of applets per workflow for free tier
PRO_TIER_MAX_RUNS = 200  # Maximum number of workflow runs per month for pro tier
ENTERPRISE_TIER_MAX_RUNS = -1  # Unlimited runs for enterprise tier

# In-memory store for rate limiting (would be replaced with a database in production)
user_runs: Dict[str, int] = {}
run_timestamps: Dict[str, Dict[str, float]] = {}

class BillingGuard(BaseHTTPMiddleware):
    """
    Middleware to enforce rate limits and feature access based on user tier.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and enforce billing rules based on the user's subscription tier.
        """
        # Get user ID from request (placeholder, would use auth in production)
        user_id = request.headers.get("X-User-ID", "anonymous")
        
        # Get user tier from environment (placeholder, would use database in production)
        user_tier = os.environ.get(f"USER_TIER_{user_id}", "free")
        
        # Check if this is a workflow run request
        if request.url.path.endswith("/run") and request.method == "POST":
            # Check if the user is within their rate limit
            if not self._check_rate_limit(user_id, user_tier):
                return JSONResponse(
                    status_code=429,
                    content={"error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "status": 429,
                        "message": "Rate limit exceeded. Upgrade to a higher tier for more workflow runs.",
                    }}
                )
            
            # For free tier, check the number of applets in the workflow
            if user_tier == "free":
                # This would be implemented based on the actual request structure
                applet_limit_response = await self._check_applet_limit(request)
                if applet_limit_response:
                    return applet_limit_response
                
        # Handle requests to premium features
        if self._is_premium_feature(request) and user_tier == "free":
            return JSONResponse(
                status_code=403,
                content={"error": {
                    "code": "FORBIDDEN",
                    "status": 403,
                    "message": "This feature is only available to Pro or Enterprise users. Please upgrade your subscription.",
                }}
            )
        
        # Process the request
        response = await call_next(request)
        
        # Update rate limit counter if this was a successful run
        if (request.url.path.endswith("/run") and 
            request.method == "POST" and 
            response.status_code == 200):
            self._increment_run_counter(user_id)
        
        return response
    
    def _check_rate_limit(self, user_id: str, tier: str) -> bool:
        """
        Check if the user is within their monthly rate limit.
        """
        # Reset counter if it's a new month
        current_month = time.strftime("%Y-%m")
        if user_id in run_timestamps and run_timestamps[user_id].get("month") != current_month:
            user_runs[user_id] = 0
            run_timestamps[user_id] = {"month": current_month}
        
        # Initialize if new user
        if user_id not in user_runs:
            user_runs[user_id] = 0
            run_timestamps[user_id] = {"month": current_month}
        
        # Check against tier limit
        if tier == "free" and user_runs[user_id] >= FREE_TIER_MAX_RUNS:
            return False
        elif tier == "pro" and user_runs[user_id] >= PRO_TIER_MAX_RUNS:
            return False
        elif tier == "enterprise" and ENTERPRISE_TIER_MAX_RUNS >= 0 and user_runs[user_id] >= ENTERPRISE_TIER_MAX_RUNS:
            return False
        
        return True
    
    def _increment_run_counter(self, user_id: str) -> None:
        """
        Increment the user's run counter.
        """
        if user_id in user_runs:
            user_runs[user_id] += 1
        else:
            user_runs[user_id] = 1
            run_timestamps[user_id] = {"month": time.strftime("%Y-%m")}
    
    async def _check_applet_limit(self, request: Request) -> Optional[JSONResponse]:
        """
        Check if the workflow has more applets than allowed for the free tier.
        This is a placeholder - in a real implementation, we would parse the request body.
        """
        body = await request.json()
        
        # Get the flow ID from the request
        flow_id = request.path_params.get("flow_id")
        
        # This is a placeholder for actual implementation
        # In a real scenario, we would fetch the flow and count its applets
        applet_count = len(body.get("nodes", []))
        
        if applet_count > FREE_TIER_MAX_APPLETS:
            return JSONResponse(
                status_code=403,
                content={"error": {
                    "code": "FORBIDDEN",
                    "status": 403,
                    "message": f"Free tier is limited to {FREE_TIER_MAX_APPLETS} applets per workflow. Upgrade to Pro for unlimited applets.",
                }}
            )
        return None
    
    def _is_premium_feature(self, request: Request) -> bool:
        """
        Check if the requested endpoint is a premium feature.
        """
        premium_endpoints = [
            "/ai/suggest",  # AI code suggestions
            "/flows/export",  # Workflow export
            "/flows/import",  # Workflow import
            "/applets/custom"  # Custom applet creation
        ]
        
        return any(request.url.path.endswith(endpoint) for endpoint in premium_endpoints)

# Function to add the middleware to the FastAPI app
def add_billing_guard(app):
    """
    Add the BillingGuard middleware to the FastAPI application.
    """
    app.add_middleware(BillingGuard)

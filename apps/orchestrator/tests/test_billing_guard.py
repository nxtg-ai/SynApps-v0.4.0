
import pytest
import os
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from apps.orchestrator.middleware.billing_guard import BillingGuard, add_billing_guard, FREE_TIER_MAX_RUNS, FREE_TIER_MAX_APPLETS

def test_billing_guard_free_tier_limits():
    app = FastAPI()
    add_billing_guard(app)
    
    @app.post("/flows/{flow_id}/run")
    async def run_flow(flow_id: str, request: Request):
        return {"status": "ok"}

    @app.get("/ai/suggest")
    async def ai_suggest():
        return {"suggestion": "some code"}

    client = TestClient(app, raise_server_exceptions=False)
    
    # Test free tier rate limit
    headers = {"X-User-ID": "test_user_free"}
    with pytest.MonkeyPatch().context() as m:
        m.setenv("USER_TIER_test_user_free", "free")
        
        # Mocking the internal state of BillingGuard if necessary, 
        # but here we can just call it many times.
        # Actually, BillingGuard uses global variables user_runs and run_timestamps.
        # This might be tricky if tests run in parallel or multiple times.
        
        from apps.orchestrator.middleware.billing_guard import user_runs
        user_runs["test_user_free"] = FREE_TIER_MAX_RUNS
        
        response = client.post("/flows/test-flow/run", json={"nodes": []}, headers=headers)
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "Rate limit exceeded" in response.json()["error"]["message"]

        # Test applet limit for free tier
        user_runs["test_user_free"] = 0
        nodes = [{"id": str(i)} for i in range(FREE_TIER_MAX_APPLETS + 1)]
        response = client.post("/flows/test-flow/run", json={"nodes": nodes}, headers=headers)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"
        assert "Free tier is limited" in response.json()["error"]["message"]

        # Test premium feature access
        response = client.get("/ai/suggest", headers=headers)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"
        assert "This feature is only available to Pro or Enterprise users" in response.json()["error"]["message"]

def test_billing_guard_pro_tier():
    app = FastAPI()
    add_billing_guard(app)
    
    @app.post("/flows/{flow_id}/run")
    async def run_flow(flow_id: str):
        return {"status": "ok"}

    @app.get("/ai/suggest")
    async def ai_suggest():
        return {"suggestion": "some code"}

    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-User-ID": "test_user_pro"}
    
    with pytest.MonkeyPatch().context() as m:
        m.setenv("USER_TIER_test_user_pro", "pro")
        
        from apps.orchestrator.middleware.billing_guard import user_runs
        user_runs["test_user_pro"] = 0
        
        # Pro should be able to run
        response = client.post("/flows/test-flow/run", json={"nodes": [1,2,3,4,5]}, headers=headers)
        assert response.status_code == 200
        
        # Pro should be able to use premium features
        response = client.get("/ai/suggest", headers=headers)
        assert response.status_code == 200

def test_billing_guard_enterprise_tier():
    app = FastAPI()
    add_billing_guard(app)
    
    @app.post("/flows/{flow_id}/run")
    async def run_flow(flow_id: str):
        return {"status": "ok"}

    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-User-ID": "test_user_ent"}
    
    with pytest.MonkeyPatch().context() as m:
        m.setenv("USER_TIER_test_user_ent", "enterprise")
        
        from apps.orchestrator.middleware.billing_guard import user_runs
        user_runs["test_user_ent"] = 1000000 
        
        # Enterprise should be unlimited (if FREE_TIER_MAX_RUNS is -1, but it's not)
        # Wait, ENTERPRISE_TIER_MAX_RUNS = -1
        response = client.post("/flows/test-flow/run", json={"nodes": []}, headers=headers)
        assert response.status_code == 200

"""
Authentication Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient, seed_roles):
    response = await client.post("/api/v1/auth/register", json={
        "full_name": "Jane Smith",
        "email": "jane@test.com",
        "password": "Secur3P@ss",
        "organization_name": "Acme Corp",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert "access_token" in data["data"]
    assert "refresh_token" in data["data"]
    assert data["data"]["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, seed_roles):
    payload = {
        "full_name": "Jane Smith",
        "email": "jane.dup@test.com",
        "password": "Secur3P@ss",
        "organization_name": "Acme Corp 1",
    }
    await client.post("/api/v1/auth/register", json=payload)

    # Second registration with same email
    payload["organization_name"] = "Acme Corp 2"
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409
    assert response.json()["error_code"] == "DUPLICATE_RESOURCE"


@pytest.mark.asyncio
async def test_register_weak_password(client: AsyncClient, seed_roles):
    response = await client.post("/api/v1/auth/register", json={
        "full_name": "Weak User",
        "email": "weak@test.com",
        "password": "password",   # no uppercase, no special char
        "organization_name": "Weak Org",
    })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, registered_user):
    response = await client.post("/api/v1/auth/login", json={
        "email": "test.admin@example.com",
        "password": "Test@123456",
    })
    assert response.status_code == 200
    assert "access_token" in response.json()["data"]


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, registered_user):
    response = await client.post("/api/v1/auth/login", json={
        "email": "test.admin@example.com",
        "password": "WrongPass!1",
    })
    assert response.status_code == 401
    assert response.json()["error_code"] == "INVALID_CREDENTIALS"


@pytest.mark.asyncio
async def test_login_unknown_email(client: AsyncClient, seed_roles):
    response = await client.post("/api/v1/auth/login", json={
        "email": "nobody@nowhere.com",
        "password": "Any@12345",
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient, auth_headers):
    response = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["email"] == "test.admin@example.com"
    assert "admin" in data["roles"]


@pytest.mark.asyncio
async def test_me_unauthenticated(client: AsyncClient):
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token(client: AsyncClient, registered_user):
    response = await client.post("/api/v1/auth/refresh", json={
        "refresh_token": registered_user["refresh_token"]
    })
    assert response.status_code == 200
    assert "access_token" in response.json()["data"]


@pytest.mark.asyncio
async def test_change_password(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/v1/auth/change-password",
        headers=auth_headers,
        json={
            "current_password": "Test@123456",
            "new_password": "NewTest@789",
        },
    )
    assert response.status_code == 200

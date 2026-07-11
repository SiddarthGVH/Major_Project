"""
Company CRUD Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_company(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/v1/companies",
        headers=auth_headers,
        json={
            "name": "TechNova Solutions",
            "website": "https://technova.io",
            "industry": "Software",
            "city": "San Francisco",
            "country": "USA",
        },
    )
    assert response.status_code == 201
    data = response.json()["data"]
    assert data["name"] == "TechNova Solutions"
    assert data["industry"] == "Software"
    return data["id"]


@pytest.mark.asyncio
async def test_create_duplicate_company(client: AsyncClient, auth_headers):
    payload = {"name": "Unique Corp"}
    await client.post("/api/v1/companies", headers=auth_headers, json=payload)
    response = await client.post("/api/v1/companies", headers=auth_headers, json=payload)
    assert response.status_code == 409
    assert response.json()["error_code"] == "DUPLICATE_RESOURCE"


@pytest.mark.asyncio
async def test_list_companies(client: AsyncClient, auth_headers):
    await client.post("/api/v1/companies", headers=auth_headers, json={"name": "List Co A"})
    await client.post("/api/v1/companies", headers=auth_headers, json={"name": "List Co B"})

    response = await client.get("/api/v1/companies", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["meta"]["total"] >= 2


@pytest.mark.asyncio
async def test_search_companies(client: AsyncClient, auth_headers):
    await client.post("/api/v1/companies", headers=auth_headers, json={"name": "SearchableXYZ Corp"})
    response = await client.get("/api/v1/companies?search=SearchableXYZ", headers=auth_headers)
    assert response.status_code == 200
    items = response.json()["data"]["data"]
    assert any("SearchableXYZ" in c["name"] for c in items)


@pytest.mark.asyncio
async def test_get_company_not_found(client: AsyncClient, auth_headers):
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await client.get(f"/api/v1/companies/{fake_id}", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_company(client: AsyncClient, auth_headers):
    create_resp = await client.post(
        "/api/v1/companies", headers=auth_headers, json={"name": "Update Me Corp"}
    )
    company_id = create_resp.json()["data"]["id"]

    update_resp = await client.put(
        f"/api/v1/companies/{company_id}",
        headers=auth_headers,
        json={"name": "Updated Corp Name", "industry": "Finance"},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["data"]["name"] == "Updated Corp Name"


@pytest.mark.asyncio
async def test_delete_company(client: AsyncClient, auth_headers):
    create_resp = await client.post(
        "/api/v1/companies", headers=auth_headers, json={"name": "Delete Me Corp"}
    )
    company_id = create_resp.json()["data"]["id"]

    delete_resp = await client.delete(f"/api/v1/companies/{company_id}", headers=auth_headers)
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/api/v1/companies/{company_id}", headers=auth_headers)
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_invalid_website_url(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/v1/companies",
        headers=auth_headers,
        json={"name": "Bad URL Corp", "website": "not-a-url"},
    )
    assert response.status_code == 422

"""
Contact CRUD Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_contact(client: AsyncClient, auth_headers):
    resp = await client.post("/api/v1/contacts", headers=auth_headers, json={
        "first_name": "Alice",
        "last_name": "Walker",
        "email": "alice.walker@example.com",
        "job_title": "CTO",
    })
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["email"] == "alice.walker@example.com"
    assert data["full_name"] == "Alice Walker"


@pytest.mark.asyncio
async def test_create_duplicate_contact_email(client: AsyncClient, auth_headers):
    payload = {
        "first_name": "Bob",
        "last_name": "Smith",
        "email": "bob.dup@example.com",
    }
    await client.post("/api/v1/contacts", headers=auth_headers, json=payload)
    resp = await client.post("/api/v1/contacts", headers=auth_headers, json=payload)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_contacts(client: AsyncClient, auth_headers):
    for i in range(3):
        await client.post("/api/v1/contacts", headers=auth_headers, json={
            "first_name": f"User{i}",
            "last_name": "Test",
            "email": f"user{i}.list@example.com",
        })
    resp = await client.get("/api/v1/contacts", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["meta"]["total"] >= 3


@pytest.mark.asyncio
async def test_update_contact(client: AsyncClient, auth_headers):
    create = await client.post("/api/v1/contacts", headers=auth_headers, json={
        "first_name": "Update",
        "last_name": "Me",
        "email": "update.me@example.com",
    })
    contact_id = create.json()["data"]["id"]

    resp = await client.put(
        f"/api/v1/contacts/{contact_id}",
        headers=auth_headers,
        json={"job_title": "VP Engineering"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["job_title"] == "VP Engineering"


@pytest.mark.asyncio
async def test_delete_contact(client: AsyncClient, auth_headers):
    create = await client.post("/api/v1/contacts", headers=auth_headers, json={
        "first_name": "Delete",
        "last_name": "Me",
        "email": "delete.contact@example.com",
    })
    contact_id = create.json()["data"]["id"]
    resp = await client.delete(f"/api/v1/contacts/{contact_id}", headers=auth_headers)
    assert resp.status_code == 204

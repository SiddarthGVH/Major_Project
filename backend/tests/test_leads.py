"""
Lead CRUD + FSM Status Transition Tests
"""
import pytest
from httpx import AsyncClient


async def _create_lead(client, headers, title="Test Lead") -> dict:
    resp = await client.post("/api/v1/leads", headers=headers, json={"title": title})
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


@pytest.mark.asyncio
async def test_create_lead(client: AsyncClient, auth_headers):
    lead = await _create_lead(client, auth_headers, "Q4 Enterprise Deal")
    assert lead["title"] == "Q4 Enterprise Deal"
    assert lead["status"] == "new"


@pytest.mark.asyncio
async def test_list_leads(client: AsyncClient, auth_headers):
    await _create_lead(client, auth_headers, "Lead A")
    await _create_lead(client, auth_headers, "Lead B")
    resp = await client.get("/api/v1/leads", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["meta"]["total"] >= 2


@pytest.mark.asyncio
async def test_lead_status_valid_transition(client: AsyncClient, auth_headers):
    lead = await _create_lead(client, auth_headers, "FSM Lead")
    lead_id = lead["id"]

    # new → contacted
    resp = await client.patch(
        f"/api/v1/leads/{lead_id}/status",
        headers=auth_headers,
        json={"status": "contacted"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "contacted"


@pytest.mark.asyncio
async def test_lead_status_invalid_transition(client: AsyncClient, auth_headers):
    lead = await _create_lead(client, auth_headers, "Invalid FSM Lead")
    lead_id = lead["id"]

    # new → won is INVALID
    resp = await client.patch(
        f"/api/v1/leads/{lead_id}/status",
        headers=auth_headers,
        json={"status": "won"},
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "BUSINESS_RULE_VIOLATION"


@pytest.mark.asyncio
async def test_lead_won_requires_close_reason(client: AsyncClient, auth_headers):
    lead = await _create_lead(client, auth_headers, "Close Reason Lead")
    lead_id = lead["id"]

    # Walk through the pipeline
    for status in ["contacted", "qualified", "proposal_sent", "negotiation"]:
        await client.patch(
            f"/api/v1/leads/{lead_id}/status",
            headers=auth_headers,
            json={"status": status},
        )

    # Won without close_reason should fail
    resp = await client.patch(
        f"/api/v1/leads/{lead_id}/status",
        headers=auth_headers,
        json={"status": "won"},
    )
    assert resp.status_code == 422

    # Won with close_reason should succeed
    resp = await client.patch(
        f"/api/v1/leads/{lead_id}/status",
        headers=auth_headers,
        json={"status": "won", "close_reason": "Signed 2-year contract"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "won"


@pytest.mark.asyncio
async def test_filter_leads_by_status(client: AsyncClient, auth_headers):
    await _create_lead(client, auth_headers, "Filter Lead")
    resp = await client.get("/api/v1/leads?status=new", headers=auth_headers)
    assert resp.status_code == 200
    items = resp.json()["data"]["data"]
    assert all(l["status"] == "new" for l in items)


@pytest.mark.asyncio
async def test_delete_lead(client: AsyncClient, auth_headers):
    lead = await _create_lead(client, auth_headers, "Delete Lead")
    resp = await client.delete(f"/api/v1/leads/{lead['id']}", headers=auth_headers)
    assert resp.status_code == 204

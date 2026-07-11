"""
Pytest Configuration & Fixtures
Uses an in-memory SQLite database for fast, isolated tests.
"""
import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.connection import get_db
from app.main import app

# ── In-memory test database ───────────────────────────────────────────────────
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestSessionFactory = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for all async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_db():
    """Create all tables once per test session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Per-test transaction that rolls back after each test."""
    async with TestSessionFactory() as session:
        try:
            yield session
        finally:
            await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client with the DB session override."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Convenience fixtures ──────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def seed_roles(db_session: AsyncSession):
    """Seed the three built-in roles + permissions into the test DB."""
    from scripts.seed import seed_permissions, seed_roles as _seed_roles
    perm_map = await seed_permissions(db_session)
    role_map = await _seed_roles(db_session, perm_map)
    await db_session.commit()
    return role_map


@pytest_asyncio.fixture
async def registered_user(client: AsyncClient, seed_roles):
    """Register a fresh user and return tokens + user data."""
    response = await client.post("/api/v1/auth/register", json={
        "full_name": "Test Admin",
        "email": "test.admin@example.com",
        "password": "Test@123456",
        "organization_name": "Test Org",
    })
    assert response.status_code == 201, response.text
    data = response.json()["data"]
    return data


@pytest_asyncio.fixture
async def auth_headers(registered_user: dict) -> dict:
    """Return Authorization headers for the registered test user."""
    return {"Authorization": f"Bearer {registered_user['access_token']}"}

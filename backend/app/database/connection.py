"""
Async SQLAlchemy 2.0 Database Connection
- Creates the async engine with connection pooling
- Provides an async session factory
- Exposes get_db() dependency for FastAPI routes
"""

from typing import AsyncGenerator
import ssl

from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _normalize_database_url(raw_url: str) -> str:
    url = make_url(raw_url)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+asyncpg")
    return str(url)


DATABASE_URL = _normalize_database_url(settings.DATABASE_URL)
url = make_url(DATABASE_URL)

# ----------------------------------------------------------
# TEMPORARY SSL WORKAROUND (Development Only)
# ----------------------------------------------------------
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
# ----------------------------------------------------------

engine_kwargs = {
    "echo": settings.DEBUG,
    "pool_pre_ping": True,
}

if url.drivername.startswith("sqlite"):
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs.update(
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_timeout=settings.DATABASE_POOL_TIMEOUT,
        pool_recycle=settings.DATABASE_POOL_RECYCLE,
    )

# -------------------- DEBUG --------------------
print("***** LOADED MY CONNECTION.PY *****")

print("\n" + "=" * 70)
print("DATABASE_URL      :", DATABASE_URL)
print("Driver            :", url.drivername)
print("Host              :", url.host)
print("Username          :", url.username)
print("SSL Context       :", ssl_context)
print("Verify Mode       :", ssl_context.verify_mode)
print("Check Hostname    :", ssl_context.check_hostname)
print("=" * 70 + "\n")

engine = create_async_engine(
    DATABASE_URL,
    connect_args={
        "ssl": ssl_context,
    },
    **engine_kwargs,
)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()




async def check_db_connection() -> bool:
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT current_user"))
            print("✅ Connected as:", result.scalar())
        return True

    except Exception as exc:
        logger.exception("Database health check failed")
        print(type(exc).__name__)
        print(exc)
        return False
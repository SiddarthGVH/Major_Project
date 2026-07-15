import asyncio

from sqlalchemy import text

from app.core.config import settings

from app.database.connection import engine

print(settings.DATABASE_URL)


async def main():
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print("Database Connected!")
            print(result.scalar())
    except Exception as e:
        print("Database Error:")
        print(type(e).__name__)
        print(e)
    finally:
        await engine.dispose()


asyncio.run(main())



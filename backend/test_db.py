<<<<<<< HEAD
﻿import asyncio
=======
import asyncio
>>>>>>> 8c70aea23112d0ec090a696619d810cd6c7fb7a2

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


<<<<<<< HEAD
asyncio.run(main())


=======
asyncio.run(main())
>>>>>>> 8c70aea23112d0ec090a696619d810cd6c7fb7a2

import asyncio
import asyncpg


async def main():
    try:
        print("Connecting to Supabase...")

        conn = await asyncpg.connect(
            host="db.xtqruznsrbsmlvzasytd.supabase.co",
            port=5432,
            user="postgres",
            password="pulseteamcrm123",
            database="postgres",
            ssl="require",
            timeout=30,
        )

        print("✅ Connected successfully!")

        version = await conn.fetchval("SELECT version();")
        print("\nPostgreSQL Version:")
        print(version)

        result = await conn.fetchval("SELECT current_database();")
        print("\nDatabase:", result)

        await conn.close()
        print("\n✅ Connection closed.")

    except Exception as e:
        print("\n❌ Connection Failed")
        print("Error Type :", type(e).__name__)
        print("Error      :", e)


if __name__ == "__main__":
    asyncio.run(main())
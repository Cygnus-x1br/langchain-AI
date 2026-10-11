import asyncio
from langchain_community.utilities import SearxSearchWrapper

async def test_searx():
    searx_host = "http://10.100.10.31:8080"
    search = SearxSearchWrapper(searx_host=searx_host, k=3)
    # Does it have ainvoke?
    try:
        res = await search.arun("teste")
        print("ARUN OK:", res)
    except Exception as e:
        print("ARUN FAILED:", e)
        try:
            res = search.run("teste")
            print("RUN OK:", res)
        except Exception as e:
            print("RUN FAILED:", e)

if __name__ == "__main__":
    asyncio.run(test_searx())

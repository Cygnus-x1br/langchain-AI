import asyncio
from classifier import classify_intent

async def test():
    last_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Crie um post simples com a imagem anexada e os textos \"Troféu Kim Mollo 2026\", \"3º lugar - 100m borbo\""},
            {"type": "image_url", "image_url": {"url": "https://example.com/test.png"}}
        ]
    }
    cat, method, raw, rew = await classify_intent(
        user_prompt="Crie um post simples com a imagem anexada e os textos \"Troféu Kim Mollo 2026\", \"3º lugar - 100m borbo\"",
        messages=[last_msg],
        last_user_msg=last_msg,
        conversation_id="test-conv",
        user_id="test-user"
    )
    print(f"CATEGORIA: {cat}")
    print(f"MÉTODO: {method}")

if __name__ == "__main__":
    asyncio.run(test())

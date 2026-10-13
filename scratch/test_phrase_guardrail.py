import asyncio
from phrase_extractor_guardrail import extract_phrases_from_text, extract_user_phrases_guardrail

async def run_tests():
    cases = [
        # 1. Aspas duplas normais
        ('Crie um post simples com a imagem anexada e os textos "Troféu Kim Mollo 2026", "3º lugar - 100m borbo"',
         ["Troféu Kim Mollo 2026", "3º lugar - 100m borbo"]),

        # 2. Aspas simples
        ("Crie um post com a foto e inclua a frase 'Campeonato Paulista 2026' e subtítulo 'Medalha de Prata'",
         ["Campeonato Paulista 2026", "Medalha de Prata"]),

        # 3. Tópicos / Bullets
        ("""Crie uma arte para o Instagram:
- Troféu Kim Mollo 2026
- 3º lugar - 100m borboleta""",
         ["Troféu Kim Mollo 2026", "3º lugar - 100m borboleta"]),

        # 4. Numeração 1. e 2.
        ("""Crie um post com os seguintes textos:
1. Troféu Kim Mollo 2026
2. 3º lugar - 100m borbo""",
         ["Troféu Kim Mollo 2026", "3º lugar - 100m borbo"]),

        # 5. Título: e Subtítulo: explícitos sem aspas
        ("Crie um post. Título: Troféu Kim Mollo 2026 | Subtítulo: 3º lugar nos 100m borbo",
         ["Troféu Kim Mollo 2026", "3º lugar nos 100m borbo"]),

        # 6. 'inclua a frase' sem aspas
        ("Crie um post para o feed e inclua a frase Troféu Kim Mollo 2026 e o subtítulo 3º lugar 100m borbo",
         ["Troféu Kim Mollo 2026", "3º lugar 100m borbo"]),

        # 7. 'com os textos' separados por vírgula sem aspas
        ("Faça uma arte para o Instagram com os textos Troféu Kim Mollo 2026, 3º lugar nos 100m borbo",
         ["Troféu Kim Mollo 2026", "3º lugar nos 100m borbo"]),
    ]

    print("=== TESTES DETERMINÍSTICOS ===")
    for i, (text, expected) in enumerate(cases, 1):
        res = extract_phrases_from_text(text)
        status = "✓ OK" if res == expected else f"✗ FAIL (obteve {res})"
        print(f"Caso {i}: {status}")
        if res != expected:
            print(f"   Esperado: {expected}")
            print(f"   Recebido: {res}")

    print("\n=== TESTE DE HISTÓRICO (REEDIÇÃO / 'FRASES SOLICITADAS') ===")
    p_orig = 'Crie um post com as frases "Troféu Kim Mollo 2026" e "3º lugar - 100m borbo"'
    p_refaz = 'Refaça o post acima utilizando as frases solicitadas mas com letras em cores de destaque'
    msgs = [
        {"role": "user", "content": p_orig},
        {"role": "assistant", "content": "post anterior"},
        {"role": "user", "content": p_refaz}
    ]
    h, s = await extract_user_phrases_guardrail(p_refaz, msgs)
    print(f"Histórico Headline: '{h}' | Subhead: '{s}'")
    assert h == "Troféu Kim Mollo 2026"
    assert s == "3º lugar - 100m borbo"
    print("✓ Histórico OK!")

if __name__ == "__main__":
    asyncio.run(run_tests())

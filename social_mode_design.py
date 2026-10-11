import os
import time
import json
import base64
from typing import Tuple, List, Dict, Any
from config import IMAGE_STORAGE_DIR

def get_base64_of_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def remove_background(input_path: str, output_path: str) -> bool:
    try:
        import rembg
        from PIL import Image
        print(f"[Design Mode] Removendo fundo de {input_path}")
        img = Image.open(input_path)
        output = rembg.remove(img)
        output.save(output_path, "PNG")
        return True
    except Exception as e:
        print(f"[Design Mode] Erro ao remover fundo: {e}")
        return False

async def render_html_to_image(html_content: str, output_path: str, width: int, height: int):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.set_content(html_content, wait_until="networkidle")
        await page.screenshot(path=output_path, type="png")
        await browser.close()

async def execute_design_mode(
    user_prompt: str,
    platform_key: str,
    briefing: Dict[str, Any],
    source_img_public_url: str,
    spec: Dict[str, Any],
    tools_called: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Motor DESIGN: Usa rembg para remover o fundo da imagem original, gera um fundo novo
    e compõe as camadas e os textos usando HTML/CSS avançado e tirando um screenshot.
    """
    start_time = time.time()
    
    # Identificar foco
    focus = briefing.get("domain", "Geral")
    
    if not source_img_public_url:
        return "❌ Erro: Modo DESIGN precisa de uma imagem base anexada.", tools_called
        
    filename = source_img_public_url.split("/")[-1]
    local_input_path = os.path.join(IMAGE_STORAGE_DIR, filename)
    cutout_filename = f"cutout_{filename}"
    local_cutout_path = os.path.join(IMAGE_STORAGE_DIR, cutout_filename)
    final_filename = f"design_{filename}"
    local_final_path = os.path.join(IMAGE_STORAGE_DIR, final_filename)
    
    # 1. Remover o Fundo (Rembg)
    has_cutout = remove_background(local_input_path, local_cutout_path)
    img_to_embed = local_cutout_path if has_cutout else local_input_path
    img_b64 = get_base64_of_image(img_to_embed)
    
    # 2. Construir o HTML
    target_w, target_h = map(int, spec["image_size"].split("x"))
    headline = briefing.get("must_include_text") or "Destaque"
    subheadline = briefing.get("topic_summary", "")[:80]
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,400;0,900;1,900&display=swap" rel="stylesheet">
        <style>
            body {{
                margin: 0; padding: 0;
                width: {target_w}px; height: {target_h}px;
                background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
                font-family: 'Montserrat', sans-serif;
                overflow: hidden; position: relative; color: white;
            }}
            .bg-texture {{
                position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                background-image: radial-gradient(circle at center, rgba(255,255,255,0.1) 0%, transparent 70%);
                z-index: 1;
            }}
            .cutout {{
                position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
                max-width: 90%; max-height: 80%; object-fit: contain;
                z-index: 2; filter: drop-shadow(0 20px 30px rgba(0,0,0,0.8));
            }}
            .overlay-text {{
                position: absolute; bottom: 0; left: 0; width: 100%;
                padding: 60px 40px; background: linear-gradient(to top, rgba(0,0,0,1) 10%, transparent);
                z-index: 3; box-sizing: border-box; text-align: center;
            }}
            .headline {{
                font-size: {int(target_h * 0.08)}px; font-weight: 900; text-transform: uppercase;
                margin: 0 0 10px 0; line-height: 1.1;
                text-shadow: 2px 2px 0 #000, -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 6px 6px 20px rgba(0,0,0,0.9);
                color: #fde047; /* Yellow highlight */
            }}
            .subheadline {{
                font-size: {int(target_h * 0.035)}px; font-weight: 400; margin: 0; color: #f8fafc;
                text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
            }}
        </style>
    </head>
    <body>
        <div class="bg-texture"></div>
        <img src="data:image/png;base64,{img_b64}" class="cutout">
        <div class="overlay-text">
            <h1 class="headline">{headline}</h1>
            <p class="subheadline">{subheadline}</p>
        </div>
    </body>
    </html>
    """
    
    # 3. Renderizar com Playwright
    print(f"[Design Mode] Renderizando HTML no Playwright ({target_w}x{target_h})...")
    try:
        await render_html_to_image(html_template, local_final_path, target_w, target_h)
        public_img_url = source_img_public_url.replace(filename, final_filename)
        status_msg = "success"
    except Exception as e:
        print(f"[Design Mode] Falha no Playwright: {e}")
        public_img_url = source_img_public_url
        status_msg = f"failed: {e}"
    
    duration = round(time.time() - start_time, 2)
    
    tools_called.append({
        "action": "social_design_generation",
        "status": status_msg,
        "duration_seconds": duration
    })
    
    out = []
    out.append(f"### 📱 Estratégia de Conteúdo: {spec['label']} (Modo DESIGN)")
    out.append(f"**Objetivo:** Recorte Automático e Composição Web")
    out.append(f"![Arte Finalizada]({public_img_url})\n")
    
    return "\n".join(out), tools_called

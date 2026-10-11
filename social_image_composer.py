import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageOps
import uuid

# Diretório para salvar fontes baixadas
FONTS_DIR = os.getenv("FONTS_DIR", "/app/fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

FONT_BOLD_URL = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"
FONT_REG_URL = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Regular.ttf"

def download_font(url: str, filename: str) -> str:
    path = os.path.join(FONTS_DIR, filename)
    if not os.path.exists(path):
        try:
            print(f"[Composer] Baixando fonte {filename}...")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"[Composer] Erro HTTP ao baixar fonte: {r.status_code}")
        except Exception as e:
            print(f"[Composer] Falha ao baixar fonte: {e}")
    return path

def get_text_dimensions(text_string, font):
    # Retorna (width, height)
    bbox = font.getbbox(text_string)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        w, _ = get_text_dimensions(" ".join(current_line), font)
        if w > max_width:
            current_line.pop()
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def compose_image(image_path: str, output_path: str, headline: str, body: str, platform_spec: dict) -> bool:
    """
    Recorta a imagem no aspect ratio ideal e aplica o texto sobre um degradê escuro.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
        
        # 1. Ajustar tamanho e Crop (Aspect Ratio)
        target_w, target_h = map(int, platform_spec["image_size"].split("x"))
        img = ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS)
        
        # 2. Criar camada de overlay (degradê)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # O degradê sobe até 60% da imagem a partir da base
        grad_height = int(target_h * 0.65)
        start_y = target_h - grad_height
        for y in range(start_y, target_h):
            # Alpha varia de 0 a 230
            alpha = int(230 * ((y - start_y) / grad_height))
            draw.line([(0, y), (target_w, y)], fill=(0, 0, 0, alpha))
            
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
        
        # 3. Preparar Fontes
        bold_path = download_font(FONT_BOLD_URL, "Montserrat-Bold.ttf")
        reg_path = download_font(FONT_REG_URL, "Montserrat-Regular.ttf")
        
        try:
            # Tamanho da fonte dinâmico (5.5% para título, 3% para corpo)
            font_head = ImageFont.truetype(bold_path, int(target_h * 0.055))
            font_body = ImageFont.truetype(reg_path, int(target_h * 0.035))
        except IOError:
            print("[Composer] Falha ao carregar fonte TTF. Usando default.")
            font_head = ImageFont.load_default()
            font_body = ImageFont.load_default()
            
        margin_x = int(target_w * 0.08)
        max_text_width = target_w - (margin_x * 2)
        
        # 4. Quebrar linhas de texto
        if headline:
            lines_head = wrap_text(headline.upper(), font_head, max_text_width)
        else:
            lines_head = []
            
        if body:
            lines_body = wrap_text(body, font_body, max_text_width)
        else:
            lines_body = []
            
        # Calcular alturas totais
        head_h = sum([get_text_dimensions(l, font_head)[1] for l in lines_head]) + (len(lines_head) * 15)
        body_h = sum([get_text_dimensions(l, font_body)[1] for l in lines_body]) + (len(lines_body) * 10)
        total_h = head_h + body_h + 30
        
        # Posicionamento Vertical (margem de 10% da base)
        current_y = target_h - total_h - int(target_h * 0.08)
        
        # Desenhar Título com Shadow/Outline sutil
        for line in lines_head:
            # Sombra
            draw.text((margin_x + 2, current_y + 2), line, font=font_head, fill=(0,0,0,180))
            # Texto
            draw.text((margin_x, current_y), line, font=font_head, fill="white")
            current_y += get_text_dimensions(line, font_head)[1] + 15
            
        current_y += 15 # Gap
        
        # Desenhar Corpo
        for line in lines_body:
            draw.text((margin_x + 1, current_y + 1), line, font=font_body, fill=(0,0,0,150))
            draw.text((margin_x, current_y), line, font=font_body, fill=(240, 240, 240, 255))
            current_y += get_text_dimensions(line, font_body)[1] + 10
            
        final_img = img.convert("RGB")
        final_img.save(output_path, "PNG")
        print(f"[Composer] Imagem final salva em {output_path}")
        return True
    except Exception as e:
        print(f"[Composer] Falha grave na composição: {e}")
        return False

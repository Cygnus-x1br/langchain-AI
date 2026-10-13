# Guia Completo: Desacoplamento de Armazenamento e Roteamento de Proxy Reverso (/images/generated)

Este documento registra a arquitetura de desacoplamento entre o **LibreChat (VM1)** e o **LangChain Orchestrator (VM2)** para entrega de mídias geradas (imagens, gráficos, vídeos) via proxy reverso unificado sob o domínio `ai.jmarc.com.br`.

---

## 1. Contexto e Motivação do Desacoplamento

### Arquitetura Anterior (Acoplada via SSH/SCP)
Na versão inicial:
- O Orquestrador (VM2: `10.100.10.31`) gerava arquivos em disco local.
- Executava comandos `scp` via rede interna para copiar cada arquivo para a VM1 (`10.100.10.30:~/LibreChat/images/generated/`).
- O LibreChat montava essa pasta e a servia publicamente através do Nginx Proxy Manager (NPM).
- **Problemas:** Dependência de chaves SSH, acoplamento de ciclo de vida entre servidores, risco de falha silenciosa de cópia e duplicação desnecessária de armazenamento em disco.

### Nova Arquitetura (Desacoplada via HTTP Direto)
- O Orquestrador (VM2) armazena os arquivos gerados em `./generated_images` (`/app/generated_images`).
- O servidor FastAPI expõe essa pasta via endpoint estático:
  ```python
  app.mount("/images/generated", StaticFiles(directory=IMAGE_STORAGE_DIR), name="images_generated")
  ```
- O Proxy Reverso (Nginx / NPM em `10.100.0.11`) intercepta as requisições que começam com `/images/generated` e as encaminha diretamente para o FastAPI na porta 8000 (`10.100.10.31:8000`).
- Todas as demais rotas de `ai.jmarc.com.br` continuam indo para o LibreChat (`10.100.10.30:3080`).

---

## 2. A Armadilha Técnica de Precedência do Nginx (`^~`)

Durante a validação prática, identificou-se um comportamento crítico do Nginx:
- Arquivos como `.mp4` e `.txt` funcionavam perfeitamente via Custom Location.
- Arquivos como `.png` e `.jpg` retornavam `404 Not Found` do LibreChat.

### Causa Raiz: Precedência de Expressões Regulares
Por padrão, blocos de Nginx que usam regex para assets estáticos:
```nginx
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ { ... }
```
possuem **maior prioridade** do que um prefixo padrão (`location /images/generated`). Dessa forma, mesmo com o caminho batendo em `/images/generated`, a terminação `.png` fazia o Nginx desviar a requisição para o host raiz (LibreChat).

### Solução: Modificador `^~`
O modificador `^~` (*prefix match with regex disabling*) informa ao Nginx:
> *"Se a URI começar com este prefixo, assuma esta rota imediatamente e NUNCA execute regras de regex."*

---

## 3. Configuração Opção 1: Via Nginx Proxy Manager (NPM - Web UI)

Se você utiliza o **Nginx Proxy Manager** (com interface gráfica na porta 81):

1. Acesse o painel do NPM e vá em **Hosts** > **Proxy Hosts**.
2. Localize o host `ai.jmarc.com.br` e clique em **Edit**.
3. Na aba **Details**:
   - **Forward Scheme:** `http`
   - **Forward Hostname / IP:** `10.100.10.30` (IP do LibreChat)
   - **Forward Port:** `3080`
   - **Cache Assets:** Recomendado desmarcar (ou garantir o uso do `^~` abaixo).
4. Na aba **Custom Locations**, clique em **Add Location**:
   - **Define location:** `^~ /images/generated` *(O prefixo `^~` é obrigatório para evitar o sequestro por regex de `.png`)*
   - **Scheme:** `http`
   - **Forward Hostname / IP:** `10.100.10.31` (IP do Orquestrador LangChain)
   - **Forward Port:** `8000`
5. (Opcional - Avançado) Se desejar cabeçalhos adicionais na Custom Location, clique na engrenagem da linha:
   ```nginx
   proxy_set_header Host $host;
   proxy_set_header X-Real-IP $remote_addr;
   proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
   proxy_set_header X-Forwarded-Proto $scheme;
   ```
6. Clique em **Save**.

---

## 4. Configuração Opção 2: Nginx Puro (Sem Interface Gráfica / Arquivo de Configuração)

Para servidores que rodam Nginx tradicional gerenciado via terminal (ex: `/etc/nginx/sites-available/ai.jmarc.com.br.conf` ou `/etc/nginx/conf.d/ai.jmarc.com.br.conf`):

### Arquivo Completo de Exemplo:
```nginx
# Redirecionamento HTTP -> HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name ai.jmarc.com.br;
    return 301 https://$host$request_uri;
}

# Servidor HTTPS com SSL Terminado
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name ai.jmarc.com.br;

    # Certificados SSL (Let's Encrypt ou Cloudflare Origin)
    ssl_certificate /etc/letsencrypt/live/ai.jmarc.com.br/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ai.jmarc.com.br/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Limite de tamanho de upload
    client_max_body_size 100M;

    # =========================================================================
    # ROTA ESPECIAL: Mídias Geradas pelo Orquestrador LangChain (VM2)
    # IMPORTANTE: O prefixo '^~' anula qualquer prioridade de regex abaixo!
    # =========================================================================
    location ^~ /images/generated {
        # Atenção: Sem barra '/' no final do proxy_pass para preservar /images/generated/...
        proxy_pass http://10.100.10.31:8000;
        
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Suporte a streaming e buffers estáticos
        proxy_buffering on;
        proxy_cache_valid 200 24h;
        expires 24h;
        add_header Cache-Control "public, no-transform";
    }

    # =========================================================================
    # REGEX DE ASSETS ESTÁTICOS DO LIBRECHAT (Não afetará /images/generated)
    # =========================================================================
    location ~* \.(ico|css|js|gif|jpeg|jpg|png|woff|ttf|otf|svg|woff2)$ {
        proxy_pass http://10.100.10.30:3080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        expires 7d;
        add_header Cache-Control "public, max-age=604800";
    }

    # =========================================================================
    # ROTA PADRÃO: Toda a Interface e API do LibreChat (VM1)
    # =========================================================================
    location / {
        proxy_pass http://10.100.10.30:3080;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts longos para SSE (Server-Sent Events) do Chat
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_buffering off;
    }
}
```

### Comandos de Validação e Aplicação no Terminal:
```bash
# 1. Testar se a sintaxe do Nginx é válida
sudo nginx -t

# 2. Se o teste retornar 'syntax is ok' e 'test is successful', aplicar:
sudo systemctl reload nginx
```

---

## 5. Como Testar e Validar a Entrega

Após configurar o NPM ou Nginx:

1. **Teste Direto no Proxy:**
   ```bash
   curl -k -I -s --resolve ai.jmarc.com.br:443:10.100.0.11 "https://ai.jmarc.com.br/images/generated/sua_imagem.png"
   ```
   *Verifique se o cabeçalho `server` responde `uvicorn` ou retorna `HTTP/2 200 OK` com `content-type: image/png`.*

2. **Teste com Query Parameter de Cache-Busting:**
   ```bash
   curl -I -s "https://ai.jmarc.com.br/images/generated/sua_imagem.png?t=123456"
   ```
   *Garante que você não está visualizando uma resposta 404 cacheada pelo Cloudflare.*

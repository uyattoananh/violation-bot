# Deploy violation-bot on Ubuntu Linux

Tested on Ubuntu 22.04 LTS / 24.04 LTS. After the four bootstrap blocks
the webapp is live at `http://YOUR-SERVER-IP:8765` with the worker
running. Each block is **copy-and-paste-into-shell**.

> **Replace every `<placeholder>` in BLOCK 3 with your own credentials.**
> Don't commit your real values to this repo.

---

## BLOCK 1 — system deps + clone (run once)

```bash
sudo apt-get update -y && sudo apt-get install -y \
    python3 python3-venv python3-pip git curl ca-certificates && \
git clone https://github.com/uyattoananh/violation-bot.git ~/violation-bot && \
cd ~/violation-bot
```

## BLOCK 2 — Python venv + dependencies (~2-5 min, downloads PyTorch CPU)

```bash
cd ~/violation-bot && \
python3 -m venv .venv-webapp && \
source .venv-webapp/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements-webapp.txt
```

## BLOCK 3 — write `.env` with credentials

> ⚠ Replace `<...>` placeholders with your real values from the
> provider consoles linked at the bottom of this file.

```bash
cat > ~/violation-bot/.env <<'EOF'
# OpenRouter (Claude Sonnet)
OPENROUTER_API_KEY=<sk-or-v1-...>
OPENROUTER_MODEL=anthropic/claude-sonnet-4.5
OPENROUTER_TITLE=violation-bot
OPENROUTER_REFERER=https://violation-bot.app

# Cloudflare R2 (photo storage, S3-compatible)
R2_ACCOUNT_ID=<32-char-hex>
R2_ACCESS_KEY_ID=<32-char-hex>
R2_SECRET_ACCESS_KEY=<64-char-hex>
R2_BUCKET=violation-bot

# Supabase (Postgres + pgvector)
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<sb_secret_...>
SUPABASE_ANON_KEY=<sb_publishable_...>

# Open-capture mode targets (created on first boot if absent)
DEFAULT_TENANT_NAME=Public Demo
DEFAULT_PROJECT_CODE=PUBLIC
DEFAULT_PROJECT_NAME=Public uploads

# Webapp (no auth in MVP — placeholder)
APP_ADMIN_PASSWORD=change-me-for-production
EOF
chmod 600 ~/violation-bot/.env
```

## BLOCK 4 — start webapp + worker (background, logs in `~/violation-bot/logs/`)

```bash
cd ~/violation-bot && \
mkdir -p logs && \
source .venv-webapp/bin/activate && \
nohup python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8765 \
    > logs/webapp.log 2>&1 & echo $! > logs/webapp.pid && \
nohup python -m webapp.worker \
    > logs/worker.log 2>&1 & echo $! > logs/worker.pid && \
sleep 3 && \
curl -sf http://127.0.0.1:8765/healthz && echo " ✔ webapp healthy" && \
echo "" && \
echo "==> open http://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}'):8765"
```

---

## Operations

```bash
# Tail live logs
tail -f ~/violation-bot/logs/webapp.log
tail -f ~/violation-bot/logs/worker.log

# Stop both processes gracefully
kill $(cat ~/violation-bot/logs/webapp.pid ~/violation-bot/logs/worker.pid)

# Open port 8765 if you have UFW firewall enabled
sudo ufw allow 8765/tcp
```

### Pull latest code + restart (one-liner)

```bash
cd ~/violation-bot && \
kill $(cat logs/webapp.pid logs/worker.pid) 2>/dev/null; sleep 1 && \
git pull && \
source .venv-webapp/bin/activate && \
pip install -q -r requirements-webapp.txt && \
nohup python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8765 \
    > logs/webapp.log 2>&1 & echo $! > logs/webapp.pid && \
nohup python -m webapp.worker > logs/worker.log 2>&1 & echo $! > logs/worker.pid && \
sleep 2 && curl -sf http://127.0.0.1:8765/healthz && echo " ✔ restarted"
```

---

## Optional — run as systemd services (survives reboot)

After this, manage the services with `sudo systemctl {start|stop|restart|status} violation-{webapp,worker}`.

```bash
USERNAME=$(whoami)
HOMEDIR=$HOME

sudo tee /etc/systemd/system/violation-webapp.service >/dev/null <<EOF
[Unit]
Description=violation-bot webapp
After=network.target

[Service]
Type=simple
User=${USERNAME}
WorkingDirectory=${HOMEDIR}/violation-bot
EnvironmentFile=${HOMEDIR}/violation-bot/.env
ExecStart=${HOMEDIR}/violation-bot/.venv-webapp/bin/python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/violation-worker.service >/dev/null <<EOF
[Unit]
Description=violation-bot classification worker
After=network.target

[Service]
Type=simple
User=${USERNAME}
WorkingDirectory=${HOMEDIR}/violation-bot
EnvironmentFile=${HOMEDIR}/violation-bot/.env
ExecStart=${HOMEDIR}/violation-bot/.venv-webapp/bin/python -m webapp.worker
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload && \
sudo systemctl enable --now violation-webapp violation-worker && \
sleep 3 && systemctl --no-pager status violation-webapp violation-worker | head -30
```

---

## Provider consoles + rotation

| Provider | Console | What to set |
|---|---|---|
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` |
| **Cloudflare R2** | [dash.cloudflare.com](https://dash.cloudflare.com) → R2 Object Storage → Manage R2 API Tokens | `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` |
| **Supabase** | [app.supabase.com](https://app.supabase.com) → Project Settings → API Keys | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY` |

### Rotation checklist (when keys leak / quarterly hygiene)

1. At each provider's console: create a **new** key/token.
2. Edit `~/violation-bot/.env` on every server with the new value.
3. Restart services:
   - **systemd**: `sudo systemctl restart violation-webapp violation-worker`
   - **manual**: `kill $(cat logs/*.pid)` then re-run BLOCK 4
4. Test: `curl http://localhost:8765/healthz`; upload a photo via the UI; confirm classification succeeds within ~10 s.
5. **Only after green verification**: delete the old key at the provider.

---

## First-boot Supabase schema

The webapp expects these tables to already exist:
`tenants`, `projects`, `photos`, `classifications`, `corrections`,
`classify_jobs`, `photo_embeddings` (with `pgvector` extension enabled).

If you're starting from a fresh Supabase project, run the schema
migration from `db/schema.sql` (or your equivalent — see project owner).
The default tenant + project rows are created automatically on first
boot if absent.

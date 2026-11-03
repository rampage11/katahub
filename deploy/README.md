# katahub — deploy

Yandex Cloud VM `mike@93.77.183.135` (`kart-technic`). Shared nginx on
`kart-technic_default` network; shared certbot. Same host as `align360`,
`f1race`, `openclaw`, `guesswhat`.

VPS layout:

```
/opt/katahub/
  backend/              # synced from ./backend/
    docker-compose.yml
    docker-compose.prod.yml
    .env                # chmod 600, from deploy/.env.prod
  data/sessions/        # uploaded session dirs
```

## First-time deploy

### 1. Secrets

```bash
cd /Users/mike/Developer/katahub/deploy
cp .env.prod.example .env.prod
$EDITOR .env.prod
chmod 600 .env.prod
```

Leave `FALLBACK_LOCAL_ONLY=1` until you have a DeepSeek key — the backend
produces a deterministic Russian verdict without hitting any API.

### 2. DNS (MANUAL)

Point `kart.katahub.ru` A → `93.77.183.135` at your registrar. Confirm:

```bash
dig +short kart.katahub.ru
```

### 3. nginx snippet (MANUAL, one-time)

```bash
scp -i ~/.ssh/id_ed25519 nginx-snippet.conf mike@93.77.183.135:/tmp/katahub-nginx-snippet.conf
ssh mike@93.77.183.135
sudo cp /opt/kart-technic/nginx/nginx.conf /opt/kart-technic/nginx/nginx.conf.bak-pre-katahub
sudo nano /opt/kart-technic/nginx/nginx.conf   # append the snippet
```

Deploy the backend FIRST (step 5) so `katahub-api` DNS resolves, then
recreate nginx so volume mounts take effect:

```bash
cd /opt/kart-technic && sudo docker compose up -d
```

### 4. Let's Encrypt (MANUAL, one-time)

```bash
ssh mike@93.77.183.135
sudo mkdir -p /var/www/katahub
sudo chown -R www-data:www-data /var/www/katahub

# Mount the webroot into nginx container: in /opt/kart-technic/docker-compose.yaml
# add under services.nginx.volumes:
#   - /var/www/katahub:/var/www/katahub:ro
cd /opt/kart-technic && sudo docker compose up -d

sudo certbot certonly --webroot -w /var/www/katahub \
    -d kart.katahub.ru --email you@example.com --agree-tos --no-eff-email

docker exec kart-technic-nginx-1 nginx -s reload
```

### 5. Deploy

```bash
cd /Users/mike/Developer/katahub/deploy
./deploy.sh
```

Idempotent — re-run any time. First build pulls `golang:1.23-alpine` and
`postgres:16-alpine`. Subsequent builds take seconds.

### 6. Smoke test

```bash
curl -s https://kart.katahub.ru/healthz    # -> {"ok":true}
open https://kart.katahub.ru/admin/        # empty list, no sessions yet
```

## Day-to-day

### Tail logs

```bash
ssh -i ~/.ssh/id_ed25519 mike@93.77.183.135 \
  'cd /opt/katahub/backend && sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api'
```

### Redeploy

```bash
cd /Users/mike/Developer/katahub/deploy && ./deploy.sh
```

### Clean restart (destructive)

```bash
./deploy.sh --purge-data    # prompts for confirmation
```

## Troubleshooting

- `nginx: host not found in upstream "katahub-api"` — deploy the backend
  first, THEN recreate the shared nginx (so it joins `kart-technic_default`
  and sees the new container).
- `/healthz` returns 502 — the api container is probably crash-looping. Check
  logs; usually a bad `DATABASE_URL` or a migration failing.
- Upload returns 400 — the zip has to contain a `manifest.json` at the top
  level (what the iOS "Поделиться сессией" button already produces).

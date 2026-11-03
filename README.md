# katahub — kart telemetry backend

Yandex Cloud VM at `93.77.183.135` (hostname `kart-technic`). Lives at
`/opt/katahub/` on the host, served from `kart.katahub.ru`.

## Scope

- Receive post-session uploads from the iPhone app (zip with manifest + IMU + HR + audio + timing).
- Expose a live WebSocket during stints for real-time dashboards and online coaching.
- Run a lap-detection + segment-analysis pipeline on uploaded sessions.
- Generate a Russian coaching verdict via DeepSeek (or a deterministic local fallback).

## Layout

```
katahub/
  backend/           # Go api + Postgres + Dockerfile + web/
    cmd/api          # main
    internal/
      api            # chi router: /healthz /sessions /live /admin
      db             # pgx pool + migration apply
      sessions       # upload + storage + metadata
      live           # in-memory pub/sub for WS→SSE fan-out
      ai             # DeepSeek verdict + local fallback
    migrations/      # SQL applied on first boot (also via init script)
    web/             # static admin UI (index.html + session.html)
    docker-compose.yml
    Dockerfile
  deploy/            # ship it: deploy.sh, prod overlay, nginx snippet, env
```

## Endpoints (MVP)

| Method | Path                       | What |
|--------|----------------------------|------|
| GET    | /healthz                   | liveness |
| POST   | /sessions                  | upload session zip (manifest.json inside) |
| GET    | /sessions                  | list sessions |
| GET    | /sessions/{id}             | single session metadata |
| POST   | /sessions/{id}/verdict     | run AI verdict on supplied Facts JSON |
| GET    | /live/{id}/publish  (WS)   | iPhone pushes live frames here |
| GET    | /live/{id}/subscribe (SSE) | admin dashboard subscribes here |
| GET    | /admin/                    | static admin UI |

## Deploy

See `deploy/README.md`. First run:

```bash
cd deploy
cp .env.prod.example .env.prod && chmod 600 .env.prod
./deploy.sh
```

## Dev loop

```bash
cd backend
docker compose up --build   # api on :8080
open http://localhost:8080/admin/
```

Hitting the API from the physical iPhone on LAN: use the Mac's LAN IP
(e.g. `http://192.168.1.42:8080`).

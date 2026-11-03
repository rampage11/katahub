#!/usr/bin/env bash
# Deploy katahub backend to the Yandex Cloud VM (kart-technic, 93.77.183.135).
#
# Idempotent: safe to re-run. Does NOT touch the pg_data volume unless invoked
# with --purge-data (which drops Postgres state — ask before using).
set -euo pipefail

SSH_HOST="mike@93.77.183.135"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -i "$SSH_KEY" -o ConnectTimeout=15)
REMOTE_DIR="/opt/katahub"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

PURGE_DATA=0
for arg in "$@"; do
  case "$arg" in
    --purge-data) PURGE_DATA=1 ;;
    *) echo "Unknown arg: $arg (supported: --purge-data)"; exit 2 ;;
  esac
done

cleanup() { echo "==> deploy script exited (status=$?)"; }
trap cleanup EXIT

echo "==> 1/8  Pre-flight sanity"
[ -f "$HERE/.env.prod" ] || { echo "ERROR: missing $HERE/.env.prod (copy from .env.prod.example)"; exit 1; }
[ -f "$SSH_KEY" ]        || { echo "ERROR: missing SSH key $SSH_KEY"; exit 1; }

if [ "$PURGE_DATA" -eq 1 ]; then
  read -r -p "!!! --purge-data will drop the Postgres volume on the VPS. Type 'yes' to confirm: " ans
  [ "$ans" = "yes" ] || { echo "aborted"; exit 1; }
fi

echo "==> 2/8  Ensuring ${REMOTE_DIR} exists on VPS"
"${SSH[@]}" "$SSH_HOST" "sudo mkdir -p ${REMOTE_DIR}/data/sessions && sudo chown -R mike:mike ${REMOTE_DIR}"

echo "==> 3/8  Syncing backend/ -> ${REMOTE_DIR}/backend/"
rsync -az --delete -e "ssh -i $SSH_KEY" \
  --exclude='.env' --exclude='*.bak' \
  "$REPO_ROOT/backend/" "$SSH_HOST:${REMOTE_DIR}/backend/"

echo "==> 4/8  Uploading .env.prod -> ${REMOTE_DIR}/backend/.env (chmod 600)"
scp -i "$SSH_KEY" "$HERE/.env.prod" "$SSH_HOST:${REMOTE_DIR}/backend/.env"
"${SSH[@]}" "$SSH_HOST" "chmod 600 ${REMOTE_DIR}/backend/.env"

echo "==> 5/8  Uploading production compose overlay"
scp -i "$SSH_KEY" "$HERE/docker-compose.prod.yml" "$SSH_HOST:${REMOTE_DIR}/backend/docker-compose.prod.yml"

if [ "$PURGE_DATA" -eq 1 ]; then
  echo "==> 5b    Purging pg_data volume (confirmed)"
  "${SSH[@]}" "$SSH_HOST" "cd ${REMOTE_DIR}/backend && sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml down -v"
fi

echo "==> 6/8  Building image + starting stack"
"${SSH[@]}" "$SSH_HOST" "cd ${REMOTE_DIR}/backend && \
  sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml build --pull api && \
  sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --remove-orphans"

echo "==> 7/8  Waiting 10s then probing /healthz"
sleep 10
"${SSH[@]}" "$SSH_HOST" "curl -fsS http://127.0.0.1:8080/healthz && echo" \
  || { echo "WARN: /healthz probe failed — api logs below"; }

echo
echo "==> api container logs (tail=40):"
"${SSH[@]}" "$SSH_HOST" "cd ${REMOTE_DIR}/backend && sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail=40 api"

echo
echo "==> 8/8  compose ps:"
"${SSH[@]}" "$SSH_HOST" "cd ${REMOTE_DIR}/backend && sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml ps"

cat <<EOF

===========================================================
katahub DEPLOY OK.

Tail logs:
  ssh -i ~/.ssh/id_ed25519 ${SSH_HOST} \\
    'cd ${REMOTE_DIR}/backend && sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api'

Once nginx + Let's Encrypt are wired up, API lives at
https://kart.katahub.ru/ (admin UI at /admin/, upload at POST /sessions).
===========================================================
EOF

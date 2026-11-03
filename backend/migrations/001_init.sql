-- Katahub schema: сессии + круги + события + вердикты.

CREATE TABLE IF NOT EXISTS sessions (
    id            UUID PRIMARY KEY,
    driver        TEXT,
    kart_number   TEXT,
    track         TEXT,
    started_at    TIMESTAMPTZ NOT NULL,
    duration_s    DOUBLE PRECISION,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    storage_path  TEXT NOT NULL,
    manifest      JSONB,
    status        TEXT NOT NULL DEFAULT 'uploaded'
);

CREATE INDEX IF NOT EXISTS sessions_started_at_idx ON sessions (started_at DESC);
CREATE INDEX IF NOT EXISTS sessions_track_idx ON sessions (track);

CREATE TABLE IF NOT EXISTS laps (
    session_id    UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    lap_number    INT NOT NULL,
    lap_time_ms   INT NOT NULL,
    sector1_ms    INT,
    sector2_ms    INT,
    sector3_ms    INT,
    position      INT,
    gap_to_lead_s DOUBLE PRECISION,
    wall_time     TIMESTAMPTZ NOT NULL,
    source        TEXT NOT NULL DEFAULT 'timing',
    PRIMARY KEY (session_id, lap_number, source)
);

CREATE TABLE IF NOT EXISTS verdicts (
    session_id    UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    facts_json    JSONB NOT NULL,
    verdict_text  TEXT NOT NULL,
    model         TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS live_events (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID NOT NULL,
    wall_time_ns  BIGINT NOT NULL,
    kind          TEXT NOT NULL,
    payload       JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS live_events_session_idx ON live_events (session_id, wall_time_ns);

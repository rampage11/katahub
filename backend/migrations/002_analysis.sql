-- Analysis JSON — весь render payload (trajectory, phases, corners, slips,
-- sectors). Хранится отдельно от verdicts, так как это не coaching, а raw
-- геометрия для SVG. Может быть большим (десятки/сотни KB).

CREATE TABLE IF NOT EXISTS analyses (
    session_id    UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    render_json   JSONB NOT NULL,
    best_lap_s    DOUBLE PRECISION,
    best_lap_num  INT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

package sessions

import (
	"archive/zip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	pool *pgxpool.Pool
	root string
}

func NewStore(pool *pgxpool.Pool, root string) *Store {
	return &Store{pool: pool, root: root}
}

// StoragePath returns the on-disk dir for a given session uuid.
func (s *Store) StoragePath(id uuid.UUID) string {
	return filepath.Join(s.root, id.String())
}

// ZipPath returns the full path to the stored payload.zip for a session.
func (s *Store) ZipPath(id uuid.UUID) string {
	return filepath.Join(s.StoragePath(id), "payload.zip")
}

// SaveVerdict upserts a verdict row for the session.
func (s *Store) SaveVerdict(ctx context.Context, id uuid.UUID, factsJSON json.RawMessage, text, model string) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO verdicts (session_id, facts_json, verdict_text, model)
		VALUES ($1,$2,$3,$4)
		ON CONFLICT (session_id) DO UPDATE SET
			facts_json = EXCLUDED.facts_json,
			verdict_text = EXCLUDED.verdict_text,
			model = EXCLUDED.model,
			created_at = now()
	`, id, factsJSON, text, model)
	return err
}

// SaveAnalysis upserts the full render JSON (trajectory, phases, corners, slips).
func (s *Store) SaveAnalysis(ctx context.Context, id uuid.UUID, renderJSON json.RawMessage, bestLapS *float64, bestLapNum *int) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO analyses (session_id, render_json, best_lap_s, best_lap_num)
		VALUES ($1,$2,$3,$4)
		ON CONFLICT (session_id) DO UPDATE SET
			render_json = EXCLUDED.render_json,
			best_lap_s = EXCLUDED.best_lap_s,
			best_lap_num = EXCLUDED.best_lap_num,
			created_at = now()
	`, id, renderJSON, bestLapS, bestLapNum)
	return err
}

// GetAnalysis loads the stored render JSON for a session, if any.
func (s *Store) GetAnalysis(ctx context.Context, id uuid.UUID) (json.RawMessage, error) {
	var raw json.RawMessage
	err := s.pool.QueryRow(ctx, `
		SELECT render_json FROM analyses WHERE session_id = $1
	`, id).Scan(&raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	return raw, err
}

// GetVerdict loads the stored verdict for a session, if any.
func (s *Store) GetVerdict(ctx context.Context, id uuid.UUID) (facts json.RawMessage, text, model string, createdAt time.Time, err error) {
	row := s.pool.QueryRow(ctx, `
		SELECT facts_json, verdict_text, model, created_at
		FROM verdicts WHERE session_id = $1
	`, id)
	err = row.Scan(&facts, &text, &model, &createdAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, "", "", time.Time{}, nil
	}
	return
}

// GetLaps loads all laps stored for a session, ordered.
func (s *Store) GetLaps(ctx context.Context, id uuid.UUID) ([]Lap, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT lap_number, lap_time_ms, sector1_ms, sector2_ms, sector3_ms, position, gap_to_lead_s, wall_time, source
		FROM laps WHERE session_id = $1 ORDER BY lap_number
	`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Lap{}
	for rows.Next() {
		var l Lap
		if err := rows.Scan(&l.Number, &l.LapTimeMs, &l.Sector1Ms, &l.Sector2Ms, &l.Sector3Ms, &l.Position, &l.GapToLeadS, &l.WallTime, &l.Source); err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, rows.Err()
}

type Lap struct {
	Number     int        `json:"n"`
	LapTimeMs  int        `json:"ms"`
	Sector1Ms  *int       `json:"s1,omitempty"`
	Sector2Ms  *int       `json:"s2,omitempty"`
	Sector3Ms  *int       `json:"s3,omitempty"`
	Position   *int       `json:"pos,omitempty"`
	GapToLeadS *float64   `json:"gap,omitempty"`
	WallTime   time.Time  `json:"wallTime"`
	Source     string     `json:"source"`
}

// InsertLaps upserts a batch of laps for a session.
func (s *Store) InsertLaps(ctx context.Context, id uuid.UUID, laps []Lap) error {
	for _, l := range laps {
		_, err := s.pool.Exec(ctx, `
			INSERT INTO laps (session_id, lap_number, lap_time_ms, sector1_ms, sector2_ms, sector3_ms, position, gap_to_lead_s, wall_time, source)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
			ON CONFLICT (session_id, lap_number, source) DO UPDATE SET
				lap_time_ms = EXCLUDED.lap_time_ms,
				sector1_ms = EXCLUDED.sector1_ms,
				sector2_ms = EXCLUDED.sector2_ms,
				sector3_ms = EXCLUDED.sector3_ms,
				position = EXCLUDED.position,
				gap_to_lead_s = EXCLUDED.gap_to_lead_s,
				wall_time = EXCLUDED.wall_time
		`, id, l.Number, l.LapTimeMs, l.Sector1Ms, l.Sector2Ms, l.Sector3Ms, l.Position, l.GapToLeadS, l.WallTime, l.Source)
		if err != nil {
			return err
		}
	}
	return nil
}

// Manifest mirrors the iOS SessionManifest shape. iOS uses `id` (UUID) and
// numeric `kartNumber`; we accept Date-as-number for startedAt/endedAt
// because Swift's JSONEncoder emits seconds-since-1970 by default.
type Manifest struct {
	ID           string      `json:"id"`
	StartedAt    jsonTime    `json:"startedAt"`
	EndedAt      *jsonTime   `json:"endedAt,omitempty"`
	Driver       string      `json:"driverId"`
	KartNumber   *int        `json:"kartNumber,omitempty"`
	TrackName    string      `json:"trackName"`
	TimingRaceID string      `json:"timingRaceID"`
}

// jsonTime decodes either an ISO-8601 string or a Swift-style
// `secondsSince1970` number (optionally with fractional seconds).
type jsonTime struct{ time.Time }

func (t *jsonTime) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		parsed, err := time.Parse(time.RFC3339Nano, s)
		if err != nil {
			return err
		}
		t.Time = parsed
		return nil
	}
	var secs float64
	if err := json.Unmarshal(b, &secs); err != nil {
		return err
	}
	sec := int64(secs)
	nsec := int64((secs - float64(sec)) * 1e9)
	t.Time = time.Unix(sec, nsec).UTC()
	return nil
}

type Session struct {
	ID         uuid.UUID       `json:"id"`
	Driver     *string         `json:"driver"`
	Kart       *string         `json:"kart"`
	Track      *string         `json:"track"`
	StartedAt  time.Time       `json:"startedAt"`
	DurationS  *float64        `json:"durationSec"`
	UploadedAt time.Time       `json:"uploadedAt"`
	Status     string          `json:"status"`
	Manifest   json.RawMessage `json:"manifest,omitempty"`
}

// Upload persists the raw session zip under <root>/<id>/payload.zip, extracts
// manifest.json, upserts a sessions row. Session id comes from the manifest
// (same UUID the iOS app writes). Re-uploads overwrite the stored zip and
// refresh metadata.
func (s *Store) Upload(ctx context.Context, body io.Reader) (*Session, error) {
	// Buffer to a temp file first so we can parse the manifest BEFORE picking
	// a destination (we want the manifest's own UUID).
	tmp, err := os.CreateTemp("", "katahub-upload-*.zip")
	if err != nil {
		return nil, err
	}
	tmpPath := tmp.Name()
	n, err := io.Copy(tmp, body)
	cerr := tmp.Close()
	if err != nil {
		os.Remove(tmpPath)
		return nil, err
	}
	if cerr != nil {
		os.Remove(tmpPath)
		return nil, cerr
	}
	if n == 0 {
		os.Remove(tmpPath)
		return nil, errors.New("empty upload")
	}

	manifest, raw, err := readManifestFromZip(tmpPath)
	if err != nil {
		os.Remove(tmpPath)
		return nil, fmt.Errorf("manifest: %w", err)
	}

	id, err := uuid.Parse(manifest.ID)
	if err != nil {
		os.Remove(tmpPath)
		return nil, fmt.Errorf("manifest.id not a UUID: %w", err)
	}
	dir := filepath.Join(s.root, id.String())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		os.Remove(tmpPath)
		return nil, err
	}
	zipPath := filepath.Join(dir, "payload.zip")
	if err := os.Rename(tmpPath, zipPath); err != nil {
		// cross-device rename falls back to copy
		if err := copyFile(tmpPath, zipPath); err != nil {
			os.Remove(tmpPath)
			return nil, err
		}
		os.Remove(tmpPath)
	}

	started := manifest.StartedAt.Time
	if started.IsZero() {
		started = time.Now().UTC()
	}
	var duration *float64
	if manifest.EndedAt != nil && !manifest.EndedAt.Time.IsZero() {
		d := manifest.EndedAt.Time.Sub(started).Seconds()
		if d > 0 {
			duration = &d
		}
	}

	sess := &Session{
		ID:         id,
		StartedAt:  started,
		UploadedAt: time.Now().UTC(),
		Status:     "uploaded",
		Manifest:   raw,
		DurationS:  duration,
	}
	if manifest.Driver != "" {
		sess.Driver = &manifest.Driver
	}
	if manifest.KartNumber != nil {
		k := fmt.Sprintf("%d", *manifest.KartNumber)
		sess.Kart = &k
	}
	if manifest.TrackName != "" {
		sess.Track = &manifest.TrackName
	}

	_, err = s.pool.Exec(ctx, `
		INSERT INTO sessions (id, driver, kart_number, track, started_at, duration_s, storage_path, manifest, status)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
		ON CONFLICT (id) DO UPDATE SET
			driver = EXCLUDED.driver,
			kart_number = EXCLUDED.kart_number,
			track = EXCLUDED.track,
			duration_s = EXCLUDED.duration_s,
			manifest = EXCLUDED.manifest,
			storage_path = EXCLUDED.storage_path,
			uploaded_at = now()
	`, sess.ID, sess.Driver, sess.Kart, sess.Track, sess.StartedAt, sess.DurationS, dir, raw, sess.Status)
	if err != nil {
		return nil, fmt.Errorf("insert session: %w", err)
	}
	return sess, nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	cerr := out.Close()
	if err != nil {
		return err
	}
	return cerr
}

func (s *Store) List(ctx context.Context, limit int) ([]Session, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id, driver, kart_number, track, started_at, duration_s, uploaded_at, status
		FROM sessions
		ORDER BY started_at DESC
		LIMIT $1
	`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Session, 0, limit)
	for rows.Next() {
		var sess Session
		if err := rows.Scan(&sess.ID, &sess.Driver, &sess.Kart, &sess.Track, &sess.StartedAt, &sess.DurationS, &sess.UploadedAt, &sess.Status); err != nil {
			return nil, err
		}
		out = append(out, sess)
	}
	return out, rows.Err()
}

func (s *Store) Get(ctx context.Context, id uuid.UUID) (*Session, error) {
	row := s.pool.QueryRow(ctx, `
		SELECT id, driver, kart_number, track, started_at, duration_s, uploaded_at, status, manifest
		FROM sessions
		WHERE id = $1
	`, id)
	var sess Session
	err := row.Scan(&sess.ID, &sess.Driver, &sess.Kart, &sess.Track, &sess.StartedAt, &sess.DurationS, &sess.UploadedAt, &sess.Status, &sess.Manifest)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &sess, nil
}

func readManifestFromZip(zipPath string) (Manifest, json.RawMessage, error) {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return Manifest{}, nil, err
	}
	defer r.Close()
	for _, f := range r.File {
		if filepath.Base(f.Name) != "manifest.json" {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return Manifest{}, nil, err
		}
		raw, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return Manifest{}, nil, err
		}
		var m Manifest
		if err := json.Unmarshal(raw, &m); err != nil {
			return Manifest{}, nil, err
		}
		return m, raw, nil
	}
	return Manifest{}, nil, errors.New("manifest.json not found in zip")
}

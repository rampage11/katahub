package db

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/jackc/pgx/v5/pgxpool"
)

var pool *pgxpool.Pool

func Init(ctx context.Context, url, migrationsDir string) error {
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		return fmt.Errorf("parse pg config: %w", err)
	}
	p, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return fmt.Errorf("new pgx pool: %w", err)
	}
	if err := p.Ping(ctx); err != nil {
		return fmt.Errorf("ping pg: %w", err)
	}
	pool = p
	log.Printf("db: connected (%s)", redact(url))

	if migrationsDir != "" {
		if err := applyMigrations(ctx, migrationsDir); err != nil {
			return fmt.Errorf("migrations: %w", err)
		}
	}
	return nil
}

func Pool() *pgxpool.Pool { return pool }

func Close() {
	if pool != nil {
		pool.Close()
	}
}

func applyMigrations(ctx context.Context, dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			log.Printf("migrations: dir %q missing — skipping", dir)
			return nil
		}
		return err
	}
	// migrations are also mounted into postgres via docker-entrypoint-initdb.d
	// on first boot — this is a belt-and-suspenders apply for existing dbs.
	files := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if filepath.Ext(e.Name()) == ".sql" {
			files = append(files, e.Name())
		}
	}
	sort.Strings(files)
	for _, f := range files {
		b, err := os.ReadFile(filepath.Join(dir, f))
		if err != nil {
			return err
		}
		if _, err := pool.Exec(ctx, string(b)); err != nil {
			return fmt.Errorf("apply %s: %w", f, err)
		}
	}
	log.Printf("db: migrations applied (%d files)", len(files))
	return nil
}

func redact(s string) string {
	// crude: hide the password segment
	at := -1
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '@' {
			at = i
			break
		}
	}
	if at < 0 {
		return s
	}
	return s[:at] + "@…"
}

package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/joho/godotenv"

	"github.com/katahub/backend/internal/ai"
	"github.com/katahub/backend/internal/api"
	"github.com/katahub/backend/internal/db"
	"github.com/katahub/backend/internal/live"
	"github.com/katahub/backend/internal/sessions"
)

func main() {
	_ = godotenv.Load()

	port := envOr("PORT", "8080")
	sessionsDir := envOr("SESSIONS_DIR", "/data/sessions")
	webDir := envOr("WEB_DIR", "/app/web")
	if _, err := os.Stat(webDir); errors.Is(err, os.ErrNotExist) {
		if alt, _ := filepath.Abs("web"); alt != "" {
			if _, err2 := os.Stat(alt); err2 == nil {
				webDir = alt
			}
		}
	}
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		log.Fatalf("mkdir sessions: %v", err)
	}

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatalf("DATABASE_URL is required")
	}
	migrationsDir := envOr("MIGRATIONS_DIR", "/app/migrations")
	if _, err := os.Stat(migrationsDir); errors.Is(err, os.ErrNotExist) {
		if alt, _ := filepath.Abs("migrations"); alt != "" {
			if _, err2 := os.Stat(alt); err2 == nil {
				migrationsDir = alt
			}
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := db.Init(ctx, dbURL, migrationsDir); err != nil {
		log.Fatalf("db init: %v", err)
	}
	defer db.Close()

	store := sessions.NewStore(db.Pool(), sessionsDir)
	bus := live.NewBus()
	verdict := ai.NewVerdict(ai.Config{
		DeepSeekKey:   os.Getenv("DEEPSEEK_API_KEY"),
		FallbackLocal: os.Getenv("FALLBACK_LOCAL_ONLY") == "1",
	})

	// Wave 2: загружаем шаблон трассы для live-refine. Если файла нет
	// или битый — продолжаем работу без него (HandleLapEvent fallback-ит
	// на raw wall-boundary). Это осознанное решение: dashboard должен
	// работать даже на треках без заранее построенного шаблона.
	templateDir := envOr("TRACKS_DIR", "/app/tracks")
	trackName := envOr("ACTIVE_TRACK", "miks")
	templatePath := filepath.Join(templateDir, trackName+".npz")
	if sz := live.MustStatTemplate(templatePath); sz > 0 {
		if tpl, err := live.LoadTemplateNPZ(templatePath); err != nil {
			log.Printf("warn: template %s load failed: %v", templatePath, err)
		} else {
			live.SetGlobalTemplate(tpl)
			log.Printf("template loaded: %s (fs=%.1f, n=%d, lap=%.2fs)",
				templatePath, tpl.Fs, len(tpl.Wz), tpl.LapDurationS())
		}
	} else {
		log.Printf("warn: template not found at %s (live-refine disabled)", templatePath)
	}
	accumulators := live.NewAccumulatorRegistry()
	lapHandlers := live.NewLapHandlerRegistry()

	router := api.NewRouter(api.Deps{
		Sessions:     store,
		Live:         bus,
		Verdict:      verdict,
		WebDir:       webDir,
		Accumulators: accumulators,
		LapHandlers:  lapHandlers,
	})

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		log.Printf("katahub api listening on :%s (sessions_dir=%s)", port, sessionsDir)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("http serve: %v", err)
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Printf("shutdown requested")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	_ = srv.Shutdown(shutdownCtx)
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

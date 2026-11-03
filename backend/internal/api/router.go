package api

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"

	"github.com/katahub/backend/internal/ai"
	"github.com/katahub/backend/internal/live"
	"github.com/katahub/backend/internal/pipeline"
	"github.com/katahub/backend/internal/sessions"
)

type Deps struct {
	Sessions *sessions.Store
	Live     *live.Bus
	Verdict  *ai.Verdict
	WebDir   string
	// Wave 2 live-compare: аккумуляторы raw IMU и per-session LapHandler-ы.
	// Инициализируются в main даже если шаблон не загружен (fallback —
	// без refine, lap writes всё равно работают через raw wall-boundary).
	Accumulators *live.AccumulatorRegistry
	LapHandlers  *live.LapHandlerRegistry
}

func NewRouter(deps Deps) http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Recoverer)

	r.Get("/healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]bool{"ok": true})
	})

	// Короткие REST endpoint'ы — защищены таймаутом 60с чтобы не висеть
	// на subprocess, зависшем coroutine и т.п. КРОМЕ /compare — он может
	// легитимно считать Python до 60с на тяжёлой сессии (handler сам
	// cancel'ит через contextWithTimeout).
	r.Group(func(r chi.Router) {
		r.Use(middleware.Timeout(60_000_000_000))
		r.Route("/sessions", func(r chi.Router) {
			r.Post("/", uploadSession(deps))
			r.Get("/", listSessions(deps))
			r.Get("/{id}", getSession(deps))
			r.Get("/{id}/analysis", getAnalysis(deps))
			r.Post("/{id}/verdict", generateVerdict(deps))
			r.Post("/{id}/replay", replaySession(deps))
		})
	})

	// Long-lived эндпоинты (WS/SSE + compare, который может длиться до 60с)
	// БЕЗ middleware.Timeout — иначе chi оборачивает ResponseWriter в
	// timeoutWriter и после дедлайна вызывает WriteHeader на уже hijacked
	// connection ("superfluous response.WriteHeader call" в логах).
	r.Get("/sessions/{id}/compare", getCompare(deps))
	r.Route("/live/{id}", func(r chi.Router) {
		r.Get("/publish", publishWS(deps))
		r.Get("/subscribe", subscribeSSE(deps))
		// Wave 2: raw watch IMU batches (binary WS). iPhone открывает лениво.
		r.Get("/watch-imu", publishWatchImuWS(deps))
	})

	// Admin UI (static).
	if deps.WebDir != "" {
		fs := http.FileServer(http.Dir(deps.WebDir))
		r.Handle("/admin/*", http.StripPrefix("/admin/", fs))
		r.Get("/", func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, "/admin/", http.StatusFound)
		})
	}

	return r
}

func uploadSession(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Clients POST raw zip (Content-Type: application/zip). 500 MB cap.
		r.Body = http.MaxBytesReader(w, r.Body, 500*1024*1024)
		defer r.Body.Close()
		sess, err := d.Sessions.Upload(r.Context(), r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// Fire-and-forget pipeline. We detach from the request context so
		// the HTTP response is not blocked on analysis.
		go func(id uuid.UUID, kart, track string) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			runPipeline(ctx, d, id, kart, track)
		}(sess.ID, deref(sess.Kart), deref(sess.Track))
		writeJSON(w, 201, sess)
	}
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// runPipeline reads the stored zip, extracts timing.json, computes Facts,
// stores laps + verdict, kicks off the Python render worker for the SVG
// dashboard, and emits a live event so any connected admin dashboard sees
// the result instantly.
func runPipeline(ctx context.Context, d Deps, id uuid.UUID, kart, track string) {
	zipPath := d.Sessions.ZipPath(id)
	facts, err := pipeline.Analyze(zipPath, kart, track)
	if err != nil {
		log.Printf("pipeline[%s]: analyze failed: %v", id, err)
		return
	}

	// Python render — отдельная тяжёлая часть (traj, фазы, slips, сектора).
	// Отправляем в независимой горутине чтобы вердикт/круги сохранились
	// даже если Python воркер упадёт.
	go func() {
		rCtx, rCancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer rCancel()
		render, err := pipeline.RenderAnalysis(rCtx, zipPath)
		if err != nil {
			log.Printf("pipeline[%s]: python render: %v", id, err)
			return
		}
		var meta struct {
			BestLap struct {
				Number int     `json:"number"`
				TimeS  float64 `json:"time_s"`
			} `json:"best_lap"`
		}
		_ = json.Unmarshal(render, &meta)
		var bestS *float64
		var bestN *int
		if meta.BestLap.TimeS > 0 {
			t := meta.BestLap.TimeS
			bestS = &t
		}
		if meta.BestLap.Number > 0 {
			n := meta.BestLap.Number
			bestN = &n
		}
		if err := d.Sessions.SaveAnalysis(rCtx, id, render, bestS, bestN); err != nil {
			log.Printf("pipeline[%s]: save analysis: %v", id, err)
			return
		}
		d.Live.Publish(live.Event{
			SessionID: id,
			Kind:      "render.done",
			Payload:   json.RawMessage(`{"ok":true}`),
		})
		log.Printf("pipeline[%s]: render saved (%d bytes)", id, len(render))
	}()

	// Авто-compare: best vs последний круг. Отдельная goroutine, non-blocking.
	// Ждём render.done чтобы иметь шанс fallback'нуться на IMU-detected laps,
	// если timing.json пустой (например, TimingClient пропустил гонку).
	go func() {
		// Собираем кандидатов на (best, last) сначала из timing (facts),
		// затем — если пусто — ждём render и берём per_lap_times.
		pickFromFacts := func() (int, int) {
			if len(facts.Laps) < 2 {
				return 0, 0
			}
			bestN, bestS, lastN := 0, 0.0, 0
			for _, l := range facts.Laps {
				if l.Lap <= 0 {
					continue
				}
				if bestN == 0 || l.TimeS < bestS {
					bestN, bestS = l.Lap, l.TimeS
				}
				if l.Lap > lastN {
					lastN = l.Lap
				}
			}
			return bestN, lastN
		}
		pickFromRender := func() (int, int) {
			rCtx, rCancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer rCancel()
			raw, err := d.Sessions.GetAnalysis(rCtx, id)
			if err != nil || len(raw) == 0 {
				return 0, 0
			}
			laps := lapsFromRender(raw, id)
			if len(laps) < 2 {
				return 0, 0
			}
			bestN, bestMs, lastN := 0, 0, 0
			for _, l := range laps {
				if bestN == 0 || l.LapTimeMs < bestMs {
					bestN, bestMs = l.Number, l.LapTimeMs
				}
				if l.Number > lastN {
					lastN = l.Number
				}
			}
			return bestN, lastN
		}
		bestN, lastN := pickFromFacts()
		if bestN == 0 || lastN == 0 {
			// Timing пустой — ждём ~5с чтобы render.done успел дописать per_lap_times
			time.Sleep(5 * time.Second)
			bestN, lastN = pickFromRender()
		}
		if bestN == 0 || lastN == 0 || bestN == lastN {
			return
		}
		cCtx, cCancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cCancel()
		cacheName := "compare_" + strconv.Itoa(bestN) + "_vs_" + strconv.Itoa(lastN) + ".html"
		cachePath := filepath.Join(d.Sessions.StoragePath(id), cacheName)
		if err := pipeline.BuildCompare(cCtx, zipPath, cachePath, bestN, lastN, false); err != nil {
			log.Printf("pipeline[%s]: auto-compare: %v", id, err)
			return
		}
		payload, _ := json.Marshal(map[string]any{"best": bestN, "other": lastN})
		d.Live.Publish(live.Event{
			SessionID: id,
			Kind:      "compare.done",
			Payload:   payload,
		})
		log.Printf("pipeline[%s]: compare %d vs %d ready", id, bestN, lastN)
	}()

	// Persist per-lap data to the laps table.
	if len(facts.Laps) > 0 {
		laps := make([]sessions.Lap, 0, len(facts.Laps))
		for _, l := range facts.Laps {
			lap := sessions.Lap{
				Number:    l.Lap,
				LapTimeMs: int(l.TimeS * 1000),
				Source:    "timing",
				WallTime:  time.Now().UTC(), // ground truth wall-time unused by pipeline atm
			}
			if l.Position > 0 {
				p := l.Position
				lap.Position = &p
			}
			if l.Sector1S > 0 {
				v := int(l.Sector1S * 1000)
				lap.Sector1Ms = &v
			}
			if l.Sector2S > 0 {
				v := int(l.Sector2S * 1000)
				lap.Sector2Ms = &v
			}
			if l.Sector3S > 0 {
				v := int(l.Sector3S * 1000)
				lap.Sector3Ms = &v
			}
			laps = append(laps, lap)
		}
		if err := d.Sessions.InsertLaps(ctx, id, laps); err != nil {
			log.Printf("pipeline[%s]: insert laps: %v", id, err)
		}
	}

	// Hand pipeline Facts to the AI layer (local or DeepSeek). The two
	// Facts structs are JSON-compatible, so we round-trip via JSON.
	aiFacts := ai.Facts{}
	factsJSON, _ := json.Marshal(facts)
	_ = json.Unmarshal(factsJSON, &aiFacts)
	res, err := d.Verdict.Analyze(ctx, aiFacts)
	if err != nil {
		log.Printf("pipeline[%s]: verdict: %v", id, err)
		return
	}
	if err := d.Sessions.SaveVerdict(ctx, id, factsJSON, res.Verdict, res.Model); err != nil {
		log.Printf("pipeline[%s]: save verdict: %v", id, err)
	}

	// Live event — admin dashboard notification.
	payload, _ := json.Marshal(map[string]any{
		"verdict": res.Verdict,
		"model":   res.Model,
		"lapCount": facts.LapCount,
		"bestLapSec": facts.BestLapS,
	})
	d.Live.Publish(live.Event{
		SessionID: id,
		Kind:      "analysis.done",
		Payload:   payload,
	})
	log.Printf("pipeline[%s]: done (%d laps, best=%.3f, model=%s)", id, facts.LapCount, facts.BestLapS, res.Model)
}

// replaySession проигрывает сохранённый watch_imu.bin сессии через live.Bus,
// чтобы dashboard показал траекторию «как будто идёт live». Query:
//   ?speed=10   — ускорение проигрывания (default 10x)
func replaySession(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(chi.URLParam(r, "id"))
		if err != nil {
			http.Error(w, "bad id", http.StatusBadRequest)
			return
		}
		speed := 10.0
		if s := r.URL.Query().Get("speed"); s != "" {
			if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
				speed = v
			}
		}
		lap := 0
		if l := r.URL.Query().Get("lap"); l != "" {
			if v, err := strconv.Atoi(l); err == nil && v > 0 {
				lap = v
			}
		}
		zipPath := d.Sessions.ZipPath(id)
		if _, err := os.Stat(zipPath); err != nil {
			http.Error(w, "no payload for session", http.StatusNotFound)
			return
		}
		// Фоновая goroutine с собственным контекстом — HTTP-запрос короткий.
		bgCtx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		_ = cancel
		// Для DTW-mapping templateIdx загружаем render JSON (если готов).
		renderJSON, _ := d.Sessions.GetAnalysis(r.Context(), id)
		count, err := live.Replay(bgCtx, d.Live, id, zipPath, live.ReplayOpts{Speed: speed, LapNumber: lap, RenderJSON: renderJSON})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, 202, map[string]any{"started": true, "events": count, "speedX": speed, "lap": lap})
	}
}

func getAnalysis(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(chi.URLParam(r, "id"))
		if err != nil {
			http.Error(w, "bad id", http.StatusBadRequest)
			return
		}
		facts, text, model, createdAt, err := d.Sessions.GetVerdict(r.Context(), id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		laps, err := d.Sessions.GetLaps(r.Context(), id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		render, err := d.Sessions.GetAnalysis(r.Context(), id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// Fallback: если в БД пусто (timing.json был пустой), берём круги из
		// render.per_lap_times — python render детектирует их по IMU даже без
		// RaceMann. UI отличает их по полю "source": "imu" vs "timing".
		if len(laps) == 0 && len(render) > 0 {
			laps = lapsFromRender(render, id)
		}
		writeJSON(w, 200, map[string]any{
			"laps":      laps,
			"facts":     facts,
			"verdict":   text,
			"model":     model,
			"createdAt": createdAt,
			"render":    render,
		})
	}
}

// getCompare on-demand генерирует HTML-страницу сравнения двух кругов.
//   ?best=N&other=M — обязательные параметры (разные целые >0)
//   ?force=1       — пересобрать даже если cache свежий
// Результат кэшируется на диск рядом с payload.zip сессии.
func getCompare(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(chi.URLParam(r, "id"))
		if err != nil {
			http.Error(w, "bad id", http.StatusBadRequest)
			return
		}
		best, _ := strconv.Atoi(r.URL.Query().Get("best"))
		other, _ := strconv.Atoi(r.URL.Query().Get("other"))
		if best <= 0 || other <= 0 || best == other {
			http.Error(w, "need best and other as different positive integers", http.StatusBadRequest)
			return
		}
		force := r.URL.Query().Get("force") == "1"

		zipPath := d.Sessions.ZipPath(id)
		if _, err := os.Stat(zipPath); err != nil {
			http.Error(w, "no payload for session", http.StatusNotFound)
			return
		}
		cacheName := "compare_" + strconv.Itoa(best) + "_vs_" + strconv.Itoa(other) + ".html"
		cachePath := filepath.Join(d.Sessions.StoragePath(id), cacheName)

		runCtx, cancel := context.WithTimeout(r.Context(), 65*time.Second)
		defer cancel()
		if err := pipeline.BuildCompare(runCtx, zipPath, cachePath, best, other, force); err != nil {
			if errors.Is(err, pipeline.ErrAlignGateFailed) {
				http.Error(w, err.Error(), http.StatusUnprocessableEntity)
				return
			}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		http.ServeFile(w, r, cachePath)
	}
}

// lapsFromRender парсит per_lap_times из render JSON. Используется как
// fallback когда db.laps пусто (timing.json без passes, но IMU-детектор
// нашёл круги).
func lapsFromRender(render json.RawMessage, _ uuid.UUID) []sessions.Lap {
	var r struct {
		PerLapTimes []struct {
			Number int     `json:"number"`
			TimeS  float64 `json:"time_s"`
			Source string  `json:"source"`
		} `json:"per_lap_times"`
	}
	if err := json.Unmarshal(render, &r); err != nil {
		return nil
	}
	out := make([]sessions.Lap, 0, len(r.PerLapTimes))
	for _, p := range r.PerLapTimes {
		if p.Number <= 0 || p.TimeS <= 0 {
			continue
		}
		src := p.Source
		if src == "" {
			src = "imu"
		}
		out = append(out, sessions.Lap{
			Number:    p.Number,
			LapTimeMs: int(p.TimeS * 1000),
			Source:    src,
			WallTime:  time.Time{},
		})
	}
	return out
}

func listSessions(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
		list, err := d.Sessions.List(r.Context(), limit)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, 200, list)
	}
}

func getSession(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(chi.URLParam(r, "id"))
		if err != nil {
			http.Error(w, "bad id", http.StatusBadRequest)
			return
		}
		sess, err := d.Sessions.Get(r.Context(), id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if sess == nil {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, 200, sess)
	}
}

func generateVerdict(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// For now the client POSTs the Facts JSON directly. Pipeline on the
		// server will replace this once Algorithm 2 lands.
		var facts ai.Facts
		if err := json.NewDecoder(r.Body).Decode(&facts); err != nil {
			http.Error(w, "bad facts: "+err.Error(), http.StatusBadRequest)
			return
		}
		res, err := d.Verdict.Analyze(r.Context(), facts)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, 200, res)
	}
}

var wsUpgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

func publishWS(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(chi.URLParam(r, "id"))
		if err != nil {
			http.Error(w, "bad id", http.StatusBadRequest)
			return
		}
		conn, err := wsUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var ev live.Event
			if err := json.Unmarshal(msg, &ev); err != nil {
				continue
			}
			ev.SessionID = id
			d.Live.Publish(ev)
			// Wave 2 hook: timing.lap → слайс IMU-буфера → compare.
			if ev.Kind == "timing.lap" && d.Accumulators != nil && d.LapHandlers != nil {
				handleTimingLap(r.Context(), d, id, ev.Payload)
			}
		}
	}
}

// handleTimingLap разбирает payload timing.lap-события и зовёт LapHandler.
// Запускается синхронно в WS-read loop (сам HandleLapEvent не блокирующий —
// compare всегда в отдельной горутине).
func handleTimingLap(ctx context.Context, d Deps, id uuid.UUID, payload json.RawMessage) {
	var ev struct {
		LapNumber int    `json:"lapNumber"`
		LapMs     int    `json:"lapMs"`
		WallTime  string `json:"wallTime"`
	}
	if err := json.Unmarshal(payload, &ev); err != nil {
		log.Printf("live[%s]: timing.lap decode: %v", id, err)
		return
	}
	if ev.LapNumber <= 0 || ev.LapMs <= 0 {
		return
	}
	wall, err := time.Parse(time.RFC3339Nano, ev.WallTime)
	if err != nil {
		wall, err = time.Parse(time.RFC3339, ev.WallTime)
		if err != nil {
			wall = time.Now().UTC()
		}
	}
	acc := d.Accumulators.Get(id)
	if acc == nil {
		// Watch IMU WS не пришёл; refine нечего делать. Можем всё равно
		// попробовать записать slice через raw bounds, но без raw IMU
		// data что писать? Выходим.
		return
	}
	// Нам нужен kart & storage dir. Из sessions store.
	sess, _ := d.Sessions.Get(ctx, id)
	kart := ""
	if sess != nil && sess.Kart != nil {
		kart = *sess.Kart
	}
	storageDir := d.Sessions.StoragePath(id)
	handler := d.LapHandlers.GetOrCreate(id, kart, storageDir, d.Live)
	tpl := live.GlobalTemplate()
	if err := handler.HandleLapEvent(ctx, acc, tpl, ev.LapNumber, ev.LapMs, wall); err != nil {
		log.Printf("live[%s]: lap %d handler: %v", id, ev.LapNumber, err)
	}
}

// publishWatchImuWS читает бинарные кадры raw watch IMU (Wave 2) и кладёт
// в per-session accumulator. Один кадр = 1-секундный батч (header 8 байт +
// N × 88 байт). Ошибки декодинга не фатальны — следующий кадр попробует.
func publishWatchImuWS(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(chi.URLParam(r, "id"))
		if err != nil {
			http.Error(w, "bad id", http.StatusBadRequest)
			return
		}
		conn, err := wsUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		acc := d.Accumulators.GetOrCreate(id)
		for {
			mtype, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if mtype != websocket.BinaryMessage {
				continue
			}
			samples, err := live.DecodeWatchImuBatch(msg)
			if err != nil {
				log.Printf("live[%s]: watch-imu decode: %v", id, err)
				continue
			}
			acc.AddBatch(samples)
		}
	}
}

func subscribeSSE(d Deps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(chi.URLParam(r, "id"))
		if err != nil {
			http.Error(w, "bad id", http.StatusBadRequest)
			return
		}
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")

		ch, cancel := d.Live.Subscribe(id)
		defer cancel()

		ctx := r.Context()
		for {
			select {
			case <-ctx.Done():
				return
			case ev, ok := <-ch:
				if !ok {
					return
				}
				b, _ := json.Marshal(ev)
				_, _ = w.Write([]byte("data: "))
				_, _ = w.Write(b)
				_, _ = w.Write([]byte("\n\n"))
				flusher.Flush()
			}
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

package live

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/katahub/backend/internal/pipeline"
)

// LapHandler — per-session состояние live-режима Wave 2:
//   • bestLapN, bestLapMs — текущий лучший круг (обновляется при каждом
//     новом более быстром лапе).
//   • savedLaps — набор {N: LiveLap} уже сохранённых на диск.
//   • mu — защищает всё перечисленное; compare запускается в goroutine
//     без удержания lock-а.
type LapHandler struct {
	mu          sync.Mutex
	sessionID   uuid.UUID
	kartNumber  string
	storageDir  string // например /data/sessions/{uuid}
	bestLapN    int
	bestLapMs   int
	savedLaps   map[int]pipeline.LiveLap
	bus         *Bus
}

func NewLapHandler(sessionID uuid.UUID, kart, storageDir string, bus *Bus) *LapHandler {
	return &LapHandler{
		sessionID:  sessionID,
		kartNumber: kart,
		storageDir: storageDir,
		savedLaps:  make(map[int]pipeline.LiveLap),
		bus:        bus,
	}
}

// LapHandlerRegistry — key=sessionID → LapHandler.
type LapHandlerRegistry struct {
	mu    sync.RWMutex
	items map[uuid.UUID]*LapHandler
}

func NewLapHandlerRegistry() *LapHandlerRegistry {
	return &LapHandlerRegistry{items: make(map[uuid.UUID]*LapHandler)}
}

func (r *LapHandlerRegistry) GetOrCreate(sessionID uuid.UUID, kart, storageDir string, bus *Bus) *LapHandler {
	r.mu.RLock()
	h, ok := r.items[sessionID]
	r.mu.RUnlock()
	if ok {
		return h
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if h, ok := r.items[sessionID]; ok {
		return h
	}
	h = NewLapHandler(sessionID, kart, storageDir, bus)
	r.items[sessionID] = h
	return h
}

func (r *LapHandlerRegistry) Get(sessionID uuid.UUID) *LapHandler {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.items[sessionID]
}

func (r *LapHandlerRegistry) Drop(sessionID uuid.UUID) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.items, sessionID)
}

// HandleLapEvent обрабатывает одно `timing.lap` событие:
//  1. RefineBoundary по accumulator'у против шаблона.
//  2. Slice [startNs, endNs], NCC-гейт.
//  3. Пишет laps/{N}.bin.
//  4. Если это самый быстрый пока круг — обновляет laps/best.bin.
//  5. Если есть best и этот круг не best — запускает compare в goroutine.
//
// Возвращает nil если accumulator пуст / шаблона нет (мягкий fallback).
// Это НЕ блокирует WS-reader: compare всегда уходит в фон.
func (h *LapHandler) HandleLapEvent(
	ctx context.Context,
	acc *Accumulator,
	tpl *Template,
	lapNumber int,
	lapMs int,
	finishWall time.Time,
) error {
	if lapNumber <= 0 || lapMs <= 0 {
		return errors.New("invalid lap event")
	}
	if acc == nil {
		return nil
	}

	// 1. RefineBoundary ±2с.
	startNs, endNs, nccAlign := RefineBoundary(finishWall, lapMs, acc, tpl, 2.0)
	// 2. Slice.
	samples := acc.Slice(startNs, endNs)
	if len(samples) < 100 {
		log.Printf("live[%s]: lap %d — only %d samples in slice, skipping",
			h.sessionID, lapNumber, len(samples))
		return nil
	}
	// 3. NCC-гейт полного круга против шаблона (не align, а похожесть геометрии).
	nccFull := NccVsTemplate(samples, tpl)
	if tpl != nil && tpl.Loaded && nccFull < 0.6 {
		log.Printf("live[%s]: lap %d NCC-fail (ncc_full=%.3f ncc_align=%.3f), skipping save",
			h.sessionID, lapNumber, nccFull, nccAlign)
		return nil
	}

	// 4. Пишем laps/{N}.bin.
	lapsDir := filepath.Join(h.storageDir, "laps")
	if err := os.MkdirAll(lapsDir, 0o755); err != nil {
		return err
	}
	binPath := filepath.Join(lapsDir, fmt.Sprintf("%d.bin", lapNumber))
	if err := WriteLapBin(binPath, samples); err != nil {
		return fmt.Errorf("write lap bin: %w", err)
	}
	lap := pipeline.LiveLap{
		LapNumber:  lapNumber,
		LapMs:      lapMs,
		FinishWall: time.Unix(0, endNs),
		BinPath:    binPath,
	}

	// 5. Регистрируем, обновляем best.
	h.mu.Lock()
	h.savedLaps[lapNumber] = lap
	isBest := false
	if h.bestLapN == 0 || lapMs < h.bestLapMs {
		h.bestLapN = lapNumber
		h.bestLapMs = lapMs
		isBest = true
		// Копируем в best.bin — это защищает от eviction и использования после
		// того как ring buffer выкинет окно.
		bestPath := filepath.Join(lapsDir, "best.bin")
		if err := copyFileRaw(binPath, bestPath); err != nil {
			log.Printf("live[%s]: copy best.bin: %v", h.sessionID, err)
		}
	}
	bestN := h.bestLapN
	otherN := lapNumber
	// Собираем snapshot кругов для compare.
	snapshot := make([]pipeline.LiveLap, 0, len(h.savedLaps))
	for _, l := range h.savedLaps {
		snapshot = append(snapshot, l)
	}
	kart := h.kartNumber
	storageDir := h.storageDir
	h.mu.Unlock()

	log.Printf("live[%s]: lap %d saved (%.2fs span, ncc_full=%.3f ncc_align=%.3f, best=%d)",
		h.sessionID, lapNumber, float64(endNs-startNs)/1e9, nccFull, nccAlign, bestN)

	// 6. Запускаем compare в отдельной горутине. isBest означает что этот круг
	// только что стал лучшим, — старый compare устарел; dashboard автоматом
	// подхватит новый compare.done.
	if bestN == otherN {
		// Только что этот круг стал лучшим; сравнивать с собой нет смысла.
		// Dashboard и так покажет лучший как эталон; следующий круг
		// триггернёт compare N vs best.
		_ = isBest
		return nil
	}

	go h.runCompare(context.Background(), snapshot, kart, storageDir, bestN, otherN)
	return nil
}

func (h *LapHandler) runCompare(ctx context.Context, snapshot []pipeline.LiveLap, kart, storageDir string, bestN, otherN int) {
	runCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	cacheName := "compare_" + strconv.Itoa(bestN) + "_vs_" + strconv.Itoa(otherN) + ".html"
	cachePath := filepath.Join(storageDir, cacheName)
	err := pipeline.BuildCompareFromLiveLaps(runCtx, h.sessionID.String(), kart, snapshot, bestN, otherN, cachePath, true)
	if err != nil {
		if errors.Is(err, pipeline.ErrAlignGateFailed) {
			log.Printf("live[%s]: live-compare %d vs %d → align gate failed", h.sessionID, bestN, otherN)
			if h.bus != nil {
				payload, _ := json.Marshal(map[string]any{
					"best": bestN, "other": otherN, "status": "align_gate_failed",
				})
				h.bus.Publish(Event{
					SessionID: h.sessionID,
					Kind:      "compare.done",
					Payload:   payload,
				})
			}
			return
		}
		log.Printf("live[%s]: live-compare %d vs %d failed: %v", h.sessionID, bestN, otherN, err)
		return
	}
	log.Printf("live[%s]: live-compare %d vs %d ready", h.sessionID, bestN, otherN)
	if h.bus != nil {
		payload, _ := json.Marshal(map[string]any{"best": bestN, "other": otherN, "source": "live"})
		h.bus.Publish(Event{
			SessionID: h.sessionID,
			Kind:      "compare.done",
			Payload:   payload,
		})
	}
}

// WriteLapBin пишет slice []imuSample в путь (создаёт dir). Один сэмпл =
// 88 байт little-endian, формат совпадает с watch_imu.bin.
func WriteLapBin(path string, samples []imuSample) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for i := range samples {
		s := samples[i]
		if err := binary.Write(f, binary.LittleEndian, &s); err != nil {
			return err
		}
	}
	return nil
}

func copyFileRaw(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

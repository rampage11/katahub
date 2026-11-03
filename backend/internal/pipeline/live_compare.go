package pipeline

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// LiveLap описывает один круг, записанный live-режимом: raw IMU + метаданные.
// Используется BuildCompareFromLiveLaps чтобы собрать минимальный session-zip
// совместимый с laps_compare.py (который ожидает watch_imu.bin + timing.json).
//
// Это один из двух архитектурных подходов (документировано в CLAUDE.md
// при деплое):
//   А) Расширить laps_compare.py принимать два .bin файла напрямую.
//   Б) Собирать синтетический zip из уже записанных live-кусков.
// Выбрано (Б): существующий Python остаётся без изменений, одно место
// склейки. Trade-off: пишем временные файлы на каждую новую compare, но
// это дешёво рядом с CPU-cost самого compare.
type LiveLap struct {
	LapNumber int
	LapMs     int
	FinishWall time.Time
	BinPath   string // путь к N.bin с 88-байтными IMUSampleBinary
}

// BuildCompareFromLiveLaps формирует синтетический session zip в temp dir
// (watch_imu.bin = concat всех laps.BinPath, timing.json = минимальный доп
// с allPasses для участвующих кругов), затем вызывает обычный BuildCompare.
// Возвращается ошибка от BuildCompare как есть (включая ErrAlignGateFailed).
//
// kartNumber нужен для timing.json "kartNumber" (laps_compare фильтрует).
func BuildCompareFromLiveLaps(
	ctx context.Context,
	sessionID string,
	kartNumber string,
	laps []LiveLap,
	bestLapN, otherLapN int,
	outPath string,
	force bool,
) error {
	if len(laps) < 2 {
		return fmt.Errorf("need ≥2 laps, have %d", len(laps))
	}
	tmpDir, err := os.MkdirTemp("", "livecompare-"+sessionID+"-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	// 1. Собираем watch_imu.bin из конкатенации кругов. Порядок по LapNumber.
	imuPath := filepath.Join(tmpDir, "watch_imu.bin")
	imuFile, err := os.Create(imuPath)
	if err != nil {
		return err
	}
	for _, lap := range laps {
		data, err := os.ReadFile(lap.BinPath)
		if err != nil {
			imuFile.Close()
			return fmt.Errorf("read lap %d bin: %w", lap.LapNumber, err)
		}
		if _, err := imuFile.Write(data); err != nil {
			imuFile.Close()
			return fmt.Errorf("write imu: %w", err)
		}
	}
	imuFile.Close()

	// 2. timing.json: достаточно allPasses с (kartNumber, lapNumber, lastLapMs,
	// wallTime). laps_compare.py использует только эти поля для определения
	// границ круга в samples. compute_session_sf_offset нуждается в
	// pass.wallTime совпадающем с tail samples.ts_ns — что у нас по
	// построению: мы писали lap.bin как slice [startNs, finishNs] где
	// finishNs == FinishWall.UnixNano().
	type pass struct {
		LapNumber  int    `json:"lapNumber"`
		LastLapMs  int    `json:"lastLapMs"`
		KartNumber string `json:"kartNumber"`
		WallTime   string `json:"wallTime"`
	}
	type timingDoc struct {
		KartNumber string `json:"kartNumber"`
		AllPasses  []pass `json:"allPasses"`
		Laps       []pass `json:"laps"`
	}
	doc := timingDoc{KartNumber: kartNumber}
	for _, l := range laps {
		doc.AllPasses = append(doc.AllPasses, pass{
			LapNumber:  l.LapNumber,
			LastLapMs:  l.LapMs,
			KartNumber: kartNumber,
			WallTime:   l.FinishWall.UTC().Format(time.RFC3339Nano),
		})
	}
	doc.Laps = doc.AllPasses
	timingJSON, _ := json.MarshalIndent(doc, "", "  ")
	if err := os.WriteFile(filepath.Join(tmpDir, "timing.json"), timingJSON, 0o644); err != nil {
		return err
	}

	// 3. manifest.json — минимальный. SessionID должен быть UUID; передаём
	// как есть. laps_compare.py manifest не читает, но Store.Upload — да.
	// Этот zip мы не загружаем в стор, только скармливаем в BuildCompare.
	manifest := map[string]any{
		"id":           sessionID,
		"startedAt":    time.Now().UTC().Format(time.RFC3339),
		"endedAt":      time.Now().UTC().Format(time.RFC3339),
		"driverId":     sessionID,
		"kartNumber":   atoiOrZero(kartNumber),
		"trackName":    "",
		"timingRaceID": "",
		"sources":      []any{},
	}
	manifestJSON, _ := json.MarshalIndent(manifest, "", "  ")
	if err := os.WriteFile(filepath.Join(tmpDir, "manifest.json"), manifestJSON, 0o644); err != nil {
		return err
	}

	// 4. Zip его. laps_compare.py принимает zip.
	zipPath := filepath.Join(tmpDir, "payload.zip")
	if err := zipDirectory(zipPath, tmpDir); err != nil {
		return err
	}

	// 5. Делегируем в обычный BuildCompare.
	return BuildCompare(ctx, zipPath, outPath, bestLapN, otherLapN, force)
}

// zipDirectory собирает все .bin/.json файлы из dir в zip.
func zipDirectory(zipPath, dir string) error {
	f, err := os.Create(zipPath)
	if err != nil {
		return err
	}
	defer f.Close()
	zw := zip.NewWriter(f)
	defer zw.Close()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if filepath.Ext(e.Name()) == ".zip" {
			continue
		}
		src := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(src)
		if err != nil {
			return err
		}
		w, err := zw.Create(e.Name())
		if err != nil {
			return err
		}
		if _, err := w.Write(data); err != nil {
			return err
		}
	}
	return nil
}

func atoiOrZero(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + int(c-'0')
	}
	return n
}


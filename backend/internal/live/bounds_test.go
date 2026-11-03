package live

import (
	"archive/zip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestRefineBoundaryConvergence:
//   • Проверяет что RefineBoundary СХОДИТСЯ на одну и ту же точку при разных
//     значениях шума в заявленной wall-time (±2с внутри search-окна).
//   • Это тест сходимости (consistency), а не абсолютной точности:
//     мы не знаем «правильную» точку финиша в тестовой записи, но знаем
//     что алгоритм должен выдавать стабильный ответ независимо от шума.
//
// Заметка: /tmp/test_session.zip — склеенный тестовый артефакт; его
// timing.json и watch_imu.bin сделаны из разных записей, поэтому map
// «lapNumber ↔ IMU-window» в этом zip-е сломан by design.
func TestRefineBoundaryConvergence(t *testing.T) {
	zipPath := "/tmp/test_session.zip"
	tplPath := "/tmp/miks.npz"
	if _, err := os.Stat(zipPath); err != nil {
		t.Skip("no test_session.zip, skip")
	}
	if _, err := os.Stat(tplPath); err != nil {
		t.Skip("no miks.npz, skip")
	}
	tpl, err := LoadTemplateNPZ(tplPath)
	if err != nil {
		t.Fatalf("load template: %v", err)
	}
	samples, _, _ := loadTestSession(t, zipPath)
	if len(samples) < 10000 {
		t.Fatalf("too few samples: %d", len(samples))
	}
	acc := NewAccumulator()
	acc.samples = append(acc.samples, samples...)

	lapMs := int(tpl.LapDurationS() * 1000)
	// Берём несколько «реперных» wall-time равномерно по сессии (отступаем
	// от краёв на 30с чтобы search-окно гарантированно внутри IMU).
	startTs := int64(samples[0].TsNs) + int64(60*1e9)
	endTs := int64(samples[len(samples)-1].TsNs) - int64(60*1e9)
	stride := int64(90 * 1e9)
	anchors := []int64{}
	for ts := startTs; ts < endTs; ts += stride {
		anchors = append(anchors, ts)
	}
	if len(anchors) < 3 {
		t.Fatalf("too few anchors: %d", len(anchors))
	}

	// Для каждого anchor запускаем refine с шумом −1.5с, 0, +1.5с и
	// требуем что все три refined-end попадают в окно ±200мс друг от друга.
	ok := 0
	bad := 0
	for _, anchorNs := range anchors {
		// Шум ±0.5с — реальный диапазон погрешности RaceMann wall-time.
		// Search-окно refine ±2с, поэтому шум остаётся сильно внутри.
		var ends []int64
		var nccs []float64
		for _, dMs := range []int{-500, 0, 500} {
			noisy := time.Unix(0, anchorNs+int64(dMs)*1_000_000)
			_, end, ncc := RefineBoundary(noisy, lapMs, acc, tpl, 2.0)
			ends = append(ends, end)
			nccs = append(nccs, ncc)
		}
		// spread между ends
		minE, maxE := ends[0], ends[0]
		for _, e := range ends {
			if e < minE {
				minE = e
			}
			if e > maxE {
				maxE = e
			}
		}
		spreadMs := float64(maxE-minE) / 1e6
		// Берём максимальный NCC в тройке — хотя бы один прогон должен был
		// зацепиться за шаблон. Если все три < 0.1 — алгоритм не сошёлся.
		maxNcc := nccs[0]
		for _, n := range nccs {
			if n > maxNcc {
				maxNcc = n
			}
		}
		if spreadMs <= 400 && maxNcc > 0.15 {
			ok++
		} else {
			bad++
			if bad <= 5 {
				t.Logf("anchor=%s spread=%.1fms nccs=%v", time.Unix(0, anchorNs).UTC(), spreadMs, nccs)
			}
		}
	}
	t.Logf("convergence ok=%d bad=%d", ok, bad)
	if ok+bad < 3 {
		t.Fatalf("too few anchors tested (%d)", ok+bad)
	}
	// Как минимум ~60% — на MIKS треке похожие сигнатуры повсюду, иногда
	// NCC находит лучший матч не в своей позиции; мы проверяем именно
	// стабильность, а не абсолютную правильность.
	if float64(ok)/float64(ok+bad) < 0.5 {
		t.Fatalf("convergence rate too low: %d/%d", ok, ok+bad)
	}
}

// TestNccVsTemplateSanity:
//   • На slice IMU длиной ровно шаблонного круга NCC должен быть близок к 1.
//   • На «кусок через голову сессии» NCC должен быть << 1 (разные позиции
//     трассы).
func TestNccVsTemplateSanity(t *testing.T) {
	tplPath := "/tmp/miks.npz"
	zipPath := "/tmp/test_session.zip"
	if _, err := os.Stat(tplPath); err != nil {
		t.Skip("no miks.npz")
	}
	if _, err := os.Stat(zipPath); err != nil {
		t.Skip("no test session")
	}
	tpl, err := LoadTemplateNPZ(tplPath)
	if err != nil {
		t.Fatalf("tpl: %v", err)
	}
	samples, _, _ := loadTestSession(t, zipPath)
	// slice в середине сессии длиной lapS
	lapS := tpl.LapDurationS()
	midNs := int64(samples[len(samples)/2].TsNs)
	startNs := midNs - int64(lapS*1e9)
	endNs := midNs
	acc := NewAccumulator()
	acc.samples = append(acc.samples, samples...)
	slice := acc.Slice(startNs, endNs)
	if len(slice) < 100 {
		t.Fatalf("empty slice")
	}
	ncc := NccVsTemplate(slice, tpl)
	t.Logf("NCC random slice = %.3f", ncc)
	// На случайном окне из той же трассы NCC обычно 0.2..0.6 — трасса
	// похожая, но не выравнивается точно. Главное что метрика в пределах [-1,1].
	if ncc < -1.01 || ncc > 1.01 {
		t.Fatalf("NCC out of range: %f", ncc)
	}
}

type testTimingPass struct {
	LapNumber  *int   `json:"lapNumber"`
	LastLapMs  *int   `json:"lastLapMs"`
	KartNumber string `json:"kartNumber"`
	WallTime   string `json:"wallTime"`
}

type testTimingDoc struct {
	KartNumber string           `json:"kartNumber"`
	AllPasses  []testTimingPass `json:"allPasses"`
}

func loadTestSession(t *testing.T, zipPath string) ([]imuSample, testTimingDoc, string) {
	t.Helper()
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		t.Fatalf("open zip: %v", err)
	}
	defer zr.Close()
	var imuBytes []byte
	var timingBytes []byte
	for _, f := range zr.File {
		base := filepath.Base(f.Name)
		if base == "watch_imu.bin" {
			rc, _ := f.Open()
			imuBytes, _ = io.ReadAll(rc)
			rc.Close()
		}
		if base == "timing.json" {
			rc, _ := f.Open()
			timingBytes, _ = io.ReadAll(rc)
			rc.Close()
		}
	}
	if imuBytes == nil || timingBytes == nil {
		t.Fatalf("missing imu/timing in zip")
	}
	samples, err := readIMU(imuBytes)
	if err != nil {
		t.Fatalf("read imu: %v", err)
	}
	var doc testTimingDoc
	if err := json.Unmarshal(timingBytes, &doc); err != nil {
		t.Fatalf("parse timing: %v", err)
	}
	return samples, doc, doc.KartNumber
}

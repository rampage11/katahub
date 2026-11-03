package live

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

// Replay — tool для демонстрации: читаем watch_imu.bin из сохранённого
// zip-а сессии, downsample до 10 Гц, публикуем через Bus как если бы iOS
// стримил сейчас в реальном времени. Dashboard подписанный на SSE увидит
// это как настоящий live.
//
// Shape соответствует iOS KatahubLiveStream.publishTelemetry: aLat, aLong,
// ah, yawRateDeg, yawRad, phase, slip, lap. Для определения lap нужен
// timing.json; если его нет — lap остаётся nil и dashboard трассирует без
// сброса trail.

const imuSampleSize = 88

type imuSample struct {
	TsNs uint64
	Ax   float64
	Ay   float64
	Az   float64
	Gx   float64
	Gy   float64
	Gz   float64
	Qw   float64
	Qx   float64
	Qy   float64
	Qz   float64
}

func readIMU(raw []byte) ([]imuSample, error) {
	if len(raw)%imuSampleSize != 0 {
		return nil, fmt.Errorf("imu size %d not multiple of %d", len(raw), imuSampleSize)
	}
	n := len(raw) / imuSampleSize
	out := make([]imuSample, n)
	r := bytes.NewReader(raw)
	for i := range out {
		if err := binary.Read(r, binary.LittleEndian, &out[i]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ReplayOpts позволяет выбрать конкретный круг для проигрывания. Если
// LapNumber == 0, проигрывается вся сессия.
type ReplayOpts struct {
	Speed       float64
	LapNumber   int
	RenderJSON  []byte // от Sessions.GetAnalysis — для templateIdx mapping
}

// renderMeta — минимум из render JSON нужный replayer-у.
type renderMeta struct {
	BestLapMapping []int   `json:"best_lap_mapping"`
	TemplateFs     float64 `json:"template_fs"`
	TemplateN      int     `json:"template_n"`
}

// Replay запускает фон. горутину которая проигрывает сессию (или один круг)
// в bus. Lap-boundaries определяются по timing.json внутри zip-а.
func Replay(ctx context.Context, bus *Bus, sessionID uuid.UUID, zipPath string, opts ReplayOpts) (int, error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return 0, err
	}
	defer zr.Close()

	var imuBytes []byte
	var timingBytes []byte
	for _, f := range zr.File {
		base := filepath.Base(f.Name)
		if imuBytes == nil && base == "watch_imu.bin" {
			rc, err := f.Open()
			if err == nil {
				imuBytes, _ = io.ReadAll(rc)
				rc.Close()
			}
		}
		if base == "timing.json" {
			rc, err := f.Open()
			if err == nil {
				timingBytes, _ = io.ReadAll(rc)
				rc.Close()
			}
		}
	}
	if imuBytes == nil {
		for _, f := range zr.File {
			if filepath.Base(f.Name) == "iphone_imu.bin" {
				rc, err := f.Open()
				if err == nil {
					imuBytes, _ = io.ReadAll(rc)
					rc.Close()
				}
				break
			}
		}
	}
	if imuBytes == nil {
		return 0, errors.New("no IMU bin in session zip")
	}
	samples, err := readIMU(imuBytes)
	if err != nil {
		return 0, err
	}
	if len(samples) < 2 {
		return 0, errors.New("too few samples")
	}

	speed := opts.Speed
	if speed <= 0 {
		speed = 10
	}

	// Если задан конкретный круг — вытаскиваем окно из timing.json.
	startIdx, endIdx := 0, len(samples)
	if opts.LapNumber > 0 && timingBytes != nil {
		s, e, ok := lapWindow(timingBytes, samples, opts.LapNumber)
		if ok {
			startIdx, endIdx = s, e
			log.Printf("replay[%s]: lap %d window: samples %d..%d (%.1fs)",
				sessionID, opts.LapNumber, startIdx, endIdx,
				float64(samples[endIdx-1].TsNs-samples[startIdx].TsNs)/1e9)
		} else {
			log.Printf("replay[%s]: lap %d not found in timing; replaying full session", sessionID, opts.LapNumber)
		}
	}
	samples = samples[startIdx:endIdx]

	// BATCH processing: для replay все данные есть сразу.
	fs := 100.0
	if len(samples) > 1 {
		dur := float64(samples[len(samples)-1].TsNs-samples[0].TsNs) / 1e9
		if dur > 0 {
			fs = float64(len(samples)) / dur
		}
	}
	br := BatchProcess(samples, fs)
	log.Printf("replay[%s]: batch θ=%.0f° q_inverted=%v corr=%.3f (%d samples) · %s",
		sessionID, br.ThetaDeg, br.QInverted, br.Corr, len(samples), br.Note)

	// Разбор render JSON ради mapping — чтобы точка на дашборде двигалась
	// по карте в такт wz, а не линейно по времени.
	var meta renderMeta
	if len(opts.RenderJSON) > 0 {
		_ = json.Unmarshal(opts.RenderJSON, &meta)
		if len(meta.BestLapMapping) > 0 {
			log.Printf("replay[%s]: using DTW mapping (%d template samples, fs=%.1f)",
				sessionID, meta.TemplateN, meta.TemplateFs)
		}
	}

	publishEvery := 10
	published := 0

	go func() {
		t0 := samples[0].TsNs
		startWall := time.Now()
		for i := 0; i < len(samples); i++ {
			if ctx.Err() != nil {
				return
			}
			s := samples[i]
			offSec := float64(s.TsNs-t0) / 1e9
			target := startWall.Add(time.Duration(offSec / speed * float64(time.Second)))
			if wait := time.Until(target); wait > 0 {
				timer := time.NewTimer(wait)
				select {
				case <-ctx.Done():
					timer.Stop()
					return
				case <-timer.C:
				}
			}
			if i%publishEvery != 0 {
				continue
			}
			phase := br.Phases[i]
			aLong := br.ALongSmooth[i]
			payload := map[string]any{
				"aLong":     aLong,
				"yawRad":    br.Yaw[i],
				"phase":     phase,
				"theta_deg": br.ThetaDeg,
				"qInverted": br.QInverted,
			}
			// templateIdx — позиция на карте через DTW-mapping. Watch IMU
			// идёт на 100Hz, template mapping на fs_t (обычно 50Hz) → ds=2.
			// Lap IMU idx i → mapping idx (i/2) → templateIdx.
			if len(meta.BestLapMapping) > 0 {
				mi := i / 2
				if mi < len(meta.BestLapMapping) {
					payload["templateIdx"] = meta.BestLapMapping[mi]
					payload["templateN"] = meta.TemplateN
				}
			}
			raw, _ := json.Marshal(payload)
			bus.Publish(Event{
				SessionID: sessionID,
				WallNs:    time.Now().UnixNano(),
				Kind:      "telemetry",
				Payload:   raw,
			})
			published++
		}
		log.Printf("replay[%s]: finished, %d events published", sessionID, published)
	}()
	return len(samples) / publishEvery, nil
}

// lapWindow разбирает timing.json и возвращает [start, end) индексы
// samples которые попадают в указанный lap. Watch IMU ts — wall ns, так
// же как timing.json wallTime.
func lapWindow(timingBytes []byte, samples []imuSample, lap int) (int, int, bool) {
	type pass struct {
		LapNumber  *int    `json:"lapNumber"`
		LastLapMs  *int    `json:"lastLapMs"`
		KartNumber string  `json:"kartNumber"`
		WallTime   string  `json:"wallTime"`
	}
	var doc struct {
		KartNumber string `json:"kartNumber"`
		AllPasses  []pass `json:"allPasses"`
	}
	if err := json.Unmarshal(timingBytes, &doc); err != nil {
		return 0, 0, false
	}
	// Находим наш pass для указанного круга.
	for _, p := range doc.AllPasses {
		if p.KartNumber != doc.KartNumber {
			continue
		}
		if p.LapNumber == nil || *p.LapNumber != lap || p.LastLapMs == nil {
			continue
		}
		endTime, err := time.Parse(time.RFC3339, p.WallTime)
		if err != nil {
			return 0, 0, false
		}
		startTime := endTime.Add(-time.Duration(*p.LastLapMs) * time.Millisecond)
		startNs := uint64(startTime.UnixNano())
		endNs := uint64(endTime.UnixNano())
		s, e := -1, -1
		for i, smp := range samples {
			if s < 0 && smp.TsNs >= startNs {
				s = i
			}
			if smp.TsNs >= endNs {
				e = i
				break
			}
		}
		if s < 0 {
			return 0, 0, false
		}
		if e < 0 {
			e = len(samples)
		}
		if e-s < 10 {
			return 0, 0, false
		}
		return s, e, true
	}
	return 0, 0, false
}

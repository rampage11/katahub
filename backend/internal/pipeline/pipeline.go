package pipeline

import (
	"archive/zip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"time"
)

// Facts — всё что нужно LLM и Admin UI. Собирается из timing.json (ground
// truth RaceMann) + в будущем IMU (события и аптимумы). Поля дублируют
// internal/ai/Facts, но без импорта чтобы избежать cross-pkg deps.
type Facts struct {
	TrackName     string      `json:"track"`
	Kart          string      `json:"kart"`
	LapCount      int         `json:"lapCount"`
	BestLapS      float64     `json:"bestLapSec"`
	BestLapNumber int         `json:"bestLapNumber"`
	MedianLapS    float64     `json:"medianLapSec"`
	Position      int         `json:"finalPosition"`
	LeaderBestS   float64     `json:"leaderBestSec,omitempty"`
	Laps          []LapFact   `json:"laps"`
	Sectors       []SectorFact `json:"sectorSummary,omitempty"`
	Quality       Quality     `json:"quality"`
	Evidence      []Evidence  `json:"evidence,omitempty"`
}

type LapFact struct {
	Lap         int     `json:"n"`
	TimeS       float64 `json:"t"`
	DeltaBestS  float64 `json:"dBest,omitempty"`
	Position    int     `json:"pos,omitempty"`
	Sector1S    float64 `json:"s1,omitempty"`
	Sector2S    float64 `json:"s2,omitempty"`
	Sector3S    float64 `json:"s3,omitempty"`
}

type SectorFact struct {
	Name    string  `json:"name"`
	BestS   float64 `json:"best"`
	MedianS float64 `json:"median"`
}

// Quality — слой уверенности из v3. Каждое число 0..1.
type Quality struct {
	Timing      float64 `json:"timing"`
	LapDetect   float64 `json:"lapDetect"`
	SensorAlign float64 `json:"sensorAlign"`
	Overall     float64 `json:"overall"`
}

// Evidence — каждое утверждение имеет пруфы (v3 spec).
type Evidence struct {
	Claim      string   `json:"claim"`
	Confidence float64  `json:"confidence"`
	Refs       []string `json:"refs"`
}

// Timing — часть timing.json которую мы используем. Соответствует iOS
// TimingRecorder output: кроме laps[] (где записаны ТОЛЬКО наши круги,
// присылаемые RaceMann во время этого фрагмента записи) есть allPasses[]
// со ВСЕМИ Comp-событиями всех картов. Мы используем allPasses как
// авторитетный источник кругов пилота — отфильтровываем по kart number.
type Timing struct {
	KartNumber string       `json:"kartNumber"`
	Laps       []TimingLap  `json:"laps"`
	AllPasses  []TimingPass `json:"allPasses"`
	Flags      []TimingFlag `json:"flags"`
}

type TimingLap struct {
	LapNumber     int    `json:"lapNumber"`
	LapMs         int    `json:"lapMs"`
	BestLapMs     int    `json:"bestLapMs,omitempty"`
	Position      int    `json:"position"`
	ChaserGapMs   *int   `json:"chaserGapMs,omitempty"`
	GapToLeaderMs *int   `json:"gapToLeaderMs,omitempty"`
	RaceTimeMs    *int   `json:"raceTimeMs,omitempty"`
	WallTime      string `json:"wallTime"`
	Sector1Ms     *int   `json:"s1Ms,omitempty"`
	Sector2Ms     *int   `json:"s2Ms,omitempty"`
	Sector3Ms     *int   `json:"s3Ms,omitempty"`
}

type TimingPass struct {
	LapNumber     *int   `json:"lapNumber"`
	LastLapMs     *int   `json:"lastLapMs"`
	BestLapMs     *int   `json:"bestLapMs,omitempty"`
	Position      *int   `json:"position"`
	KartNumber    string `json:"kartNumber"`
	WallTime      string `json:"wallTime"`
	RaceTimeMs    *int   `json:"raceTimeMs,omitempty"`
	GapToLeaderMs *int   `json:"gapToLeaderMs,omitempty"`
	Sector1Ms     *int   `json:"s1Ms,omitempty"`
	Sector2Ms     *int   `json:"s2Ms,omitempty"`
	Sector3Ms     *int   `json:"s3Ms,omitempty"`
}

type TimingFlag struct {
	Flag     string `json:"flag"`
	WallTime string `json:"wallTime"`
}

// Analyze распаковывает zip, вытаскивает timing.json и manifest, собирает
// Facts. Error только если zip битый — отсутствие timing.json даёт пустой
// Facts (не ошибку).
func Analyze(zipPath, kart, trackName string) (*Facts, error) {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var timing *Timing
	for _, f := range r.File {
		if filepath.Base(f.Name) == "timing.json" {
			rc, err := f.Open()
			if err != nil {
				return nil, err
			}
			b, err := io.ReadAll(rc)
			rc.Close()
			if err != nil {
				return nil, err
			}
			var t Timing
			if err := json.Unmarshal(b, &t); err != nil {
				return nil, fmt.Errorf("decode timing: %w", err)
			}
			timing = &t
			break
		}
	}

	// Если kart не передан, пробуем взять из timing.json.
	if kart == "" && timing != nil && timing.KartNumber != "" {
		kart = timing.KartNumber
	}
	facts := &Facts{
		TrackName: trackName,
		Kart:      kart,
	}
	if timing == nil {
		facts.Quality = Quality{Timing: 0, Overall: 0}
		return facts, nil
	}
	factsFromTiming(facts, timing)
	factsFromLeader(facts, timing, kart)
	computeEvidence(facts)
	return facts, nil
}

func factsFromTiming(f *Facts, t *Timing) {
	// Авторитетный источник — allPasses, отфильтрованный по kart (мерж
	// данных ВСЕХ фрагментов записи). Это даёт полный набор кругов пилота
	// даже когда iPhone-запись развалилась и laps[] содержит один-два.
	laps := lapsFromAllPasses(t, f.Kart)

	// Если allPasses пустое — fallback на laps[].
	if len(laps) == 0 {
		for _, l := range t.Laps {
			if l.LapMs <= 0 {
				continue
			}
			lap := LapFact{
				Lap:      l.LapNumber,
				TimeS:    float64(l.LapMs) / 1000.0,
				Position: l.Position,
			}
			if l.Sector1Ms != nil {
				lap.Sector1S = float64(*l.Sector1Ms) / 1000.0
			}
			if l.Sector2Ms != nil {
				lap.Sector2S = float64(*l.Sector2Ms) / 1000.0
			}
			if l.Sector3Ms != nil {
				lap.Sector3S = float64(*l.Sector3Ms) / 1000.0
			}
			laps = append(laps, lap)
		}
	}
	sort.Slice(laps, func(i, j int) bool { return laps[i].Lap < laps[j].Lap })
	f.Laps = laps
	f.LapCount = len(laps)
	if len(laps) == 0 {
		return
	}

	// Фильтруем outlier-круги (overtake, pit) по IQR для best/median.
	cleanTimes := iqrFilter(timesOf(laps), 1.5)
	if len(cleanTimes) > 0 {
		f.BestLapS = minF(cleanTimes)
		f.MedianLapS = medianF(cleanTimes)
	} else {
		f.BestLapS = minF(timesOf(laps))
		f.MedianLapS = medianF(timesOf(laps))
	}

	// best lap number
	for _, l := range laps {
		if math.Abs(l.TimeS-f.BestLapS) < 1e-6 {
			f.BestLapNumber = l.Lap
			break
		}
	}
	// Проставляем deltaBest
	for i := range f.Laps {
		f.Laps[i].DeltaBestS = f.Laps[i].TimeS - f.BestLapS
	}
	// Финальная позиция = позиция в последнем своём круге
	if len(laps) > 0 {
		f.Position = laps[len(laps)-1].Position
	}

	// Sector summary (если данные есть)
	f.Sectors = computeSectorSummary(laps)

	// Quality
	f.Quality = Quality{
		Timing:    1.0,
		LapDetect: 1.0, // timing.json = ground truth
		Overall:   0.95,
	}
}

func lapsFromAllPasses(t *Timing, kart string) []LapFact {
	laps := make([]LapFact, 0)
	seen := map[int]bool{}
	for _, p := range t.AllPasses {
		if p.KartNumber != kart || p.LastLapMs == nil || p.LapNumber == nil {
			continue
		}
		if seen[*p.LapNumber] {
			continue
		}
		seen[*p.LapNumber] = true
		lap := LapFact{Lap: *p.LapNumber, TimeS: float64(*p.LastLapMs) / 1000.0}
		if p.Position != nil {
			lap.Position = *p.Position
		}
		if p.Sector1Ms != nil {
			lap.Sector1S = float64(*p.Sector1Ms) / 1000.0
		}
		if p.Sector2Ms != nil {
			lap.Sector2S = float64(*p.Sector2Ms) / 1000.0
		}
		if p.Sector3Ms != nil {
			lap.Sector3S = float64(*p.Sector3Ms) / 1000.0
		}
		laps = append(laps, lap)
	}
	return laps
}

func factsFromLeader(f *Facts, t *Timing, _ string) {
	// Лидер трассы — лучшее время любого карта в allPasses.
	bestByKart := map[string]float64{}
	for _, p := range t.AllPasses {
		if p.LastLapMs == nil || p.KartNumber == "" {
			continue
		}
		ms := float64(*p.LastLapMs) / 1000.0
		if ms <= 20 || ms > 180 { // sanity: 20с < круг < 3мин
			continue
		}
		if cur, ok := bestByKart[p.KartNumber]; !ok || ms < cur {
			bestByKart[p.KartNumber] = ms
		}
	}
	if len(bestByKart) == 0 {
		return
	}
	leader := math.Inf(1)
	for _, s := range bestByKart {
		if s < leader {
			leader = s
		}
	}
	f.LeaderBestS = leader
}

func computeSectorSummary(laps []LapFact) []SectorFact {
	s1, s2, s3 := []float64{}, []float64{}, []float64{}
	for _, l := range laps {
		if l.Sector1S > 0 {
			s1 = append(s1, l.Sector1S)
		}
		if l.Sector2S > 0 {
			s2 = append(s2, l.Sector2S)
		}
		if l.Sector3S > 0 {
			s3 = append(s3, l.Sector3S)
		}
	}
	out := []SectorFact{}
	if len(s1) > 0 {
		out = append(out, SectorFact{Name: "S1", BestS: minF(s1), MedianS: medianF(s1)})
	}
	if len(s2) > 0 {
		out = append(out, SectorFact{Name: "S2", BestS: minF(s2), MedianS: medianF(s2)})
	}
	if len(s3) > 0 {
		out = append(out, SectorFact{Name: "S3", BestS: minF(s3), MedianS: medianF(s3)})
	}
	return out
}

func computeEvidence(f *Facts) {
	if f.LapCount == 0 {
		return
	}
	spread := f.MedianLapS - f.BestLapS
	if spread > 0.4 {
		f.Evidence = append(f.Evidence, Evidence{
			Claim:      "inconsistent_pace",
			Confidence: math.Min(0.95, spread/0.6),
			Refs:       []string{"lap_times", "iqr_spread"},
		})
	}
	if f.LeaderBestS > 0 && f.BestLapS > 0 {
		gap := f.BestLapS - f.LeaderBestS
		if gap > 0.2 {
			f.Evidence = append(f.Evidence, Evidence{
				Claim:      "trailing_leader",
				Confidence: math.Min(0.9, gap/1.0),
				Refs:       []string{"best_lap_vs_leader_best"},
			})
		}
	}
	// Repeatable Best = median лучшей трети кругов (без outlier)
	times := timesOf(f.Laps)
	cleaned := iqrFilter(times, 1.5)
	if len(cleaned) >= 6 {
		sort.Float64s(cleaned)
		third := len(cleaned) / 3
		if third >= 2 {
			repeatable := medianF(cleaned[:third])
			if repeatable-f.BestLapS > 0.1 {
				f.Evidence = append(f.Evidence, Evidence{
					Claim:      "best_not_repeatable",
					Confidence: 0.75,
					Refs:       []string{"repeatable_best", "pb_vs_top_third_median"},
				})
			}
		}
	}
}

// -- helpers --

func timesOf(laps []LapFact) []float64 {
	out := make([]float64, len(laps))
	for i, l := range laps {
		out[i] = l.TimeS
	}
	return out
}

func iqrFilter(xs []float64, k float64) []float64 {
	if len(xs) < 4 {
		return xs
	}
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	q1 := percentile(sorted, 0.25)
	q3 := percentile(sorted, 0.75)
	iqr := q3 - q1
	lo, hi := q1-k*iqr, q3+k*iqr
	out := make([]float64, 0, len(xs))
	for _, v := range xs {
		if v >= lo && v <= hi {
			out = append(out, v)
		}
	}
	return out
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := p * float64(len(sorted)-1)
	lo, hi := int(math.Floor(i)), int(math.Ceil(i))
	if lo == hi {
		return sorted[lo]
	}
	w := i - float64(lo)
	return sorted[lo]*(1-w) + sorted[hi]*w
}

func minF(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	m := xs[0]
	for _, v := range xs[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func medianF(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[len(s)/2]
}

// ParseWallTime парсит ISO-8601, в идеале с миллисекундами. Returns zero
// time on parse error (callers should check).
func ParseWallTime(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05.000Z"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t
		}
	}
	return time.Time{}
}

// Unused but keeps imports stable in case we add numeric parsing later.
var _ = strconv.Itoa
var _ = errors.New

package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

type Config struct {
	DeepSeekKey   string
	FallbackLocal bool
}

type Verdict struct {
	cfg    Config
	client *http.Client
}

func NewVerdict(cfg Config) *Verdict {
	return &Verdict{cfg: cfg, client: &http.Client{Timeout: 45 * time.Second}}
}

// Facts is a structured summary built from pipeline output. Keep narrow —
// the LLM gets numbers, not raw IMU. Fields match internal/pipeline.Facts
// so an upload-time pipeline can hand its JSON straight into Analyze().
type Facts struct {
	TrackName     string       `json:"track"`
	Kart          string       `json:"kart"`
	LapCount      int          `json:"lapCount"`
	BestLapS      float64      `json:"bestLapSec"`
	BestLapNumber int          `json:"bestLapNumber"`
	MedianLapS    float64      `json:"medianLapSec"`
	Position      int          `json:"finalPosition"`
	LeaderBestS   float64      `json:"leaderBestSec,omitempty"`
	Laps          []LapFact    `json:"laps"`
	Slips         []SlipFact   `json:"slipEvents,omitempty"`
	Sectors       []SectorFact `json:"sectorSummary,omitempty"`
	Quality       Quality      `json:"quality,omitempty"`
	Evidence      []Evidence   `json:"evidence,omitempty"`
	HRMean        float64      `json:"hrMean,omitempty"`
	HRMax         float64      `json:"hrMax,omitempty"`
}

type LapFact struct {
	Lap        int     `json:"n"`
	TimeS      float64 `json:"t"`
	DeltaBestS float64 `json:"dBest,omitempty"`
	Position   int     `json:"pos,omitempty"`
	Sector1S   float64 `json:"s1,omitempty"`
	Sector2S   float64 `json:"s2,omitempty"`
	Sector3S   float64 `json:"s3,omitempty"`
}

type SlipFact struct {
	Lap      int     `json:"n"`
	SectorS  float64 `json:"s"`
	Strength float64 `json:"excess"`
}

type SectorFact struct {
	Name       string  `json:"name"`
	BestS      float64 `json:"best"`
	MedianS    float64 `json:"median"`
	ReferenceS float64 `json:"ref,omitempty"`
}

type Quality struct {
	Timing      float64 `json:"timing"`
	LapDetect   float64 `json:"lapDetect"`
	SensorAlign float64 `json:"sensorAlign"`
	Overall     float64 `json:"overall"`
}

type Evidence struct {
	Claim      string   `json:"claim"`
	Confidence float64  `json:"confidence"`
	Refs       []string `json:"refs"`
}

type Result struct {
	Model   string `json:"model"`
	Verdict string `json:"verdict"`
}

func (v *Verdict) Analyze(ctx context.Context, f Facts) (*Result, error) {
	if v.cfg.FallbackLocal || v.cfg.DeepSeekKey == "" {
		return v.local(f), nil
	}
	prompt := buildPrompt(f)
	body := map[string]any{
		"model":       "deepseek-chat",
		"temperature": 0.4,
		"messages": []map[string]string{
			{"role": "system", "content": "Ты тренер по картингу. Отвечай по-русски, кратко, по делу. 4-6 предложений. Укажи где теряется время и 2 конкретных совета."},
			{"role": "user", "content": prompt},
		},
	}
	buf, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.deepseek.com/v1/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+v.cfg.DeepSeekKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := v.client.Do(req)
	if err != nil {
		log.Printf("deepseek: request failed: %v — falling back to local verdict", err)
		return v.local(f), nil
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		log.Printf("deepseek: HTTP %d body=%s — falling back", resp.StatusCode, string(raw))
		return v.local(f), nil
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("decode deepseek: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return nil, errors.New("deepseek: empty choices")
	}
	return &Result{Model: "deepseek-chat", Verdict: parsed.Choices[0].Message.Content}, nil
}

func (v *Verdict) local(f Facts) *Result {
	spread := 0.0
	if f.MedianLapS > 0 {
		spread = f.MedianLapS - f.BestLapS
	}
	gap := 0.0
	if f.LeaderBestS > 0 && f.BestLapS > 0 {
		gap = f.BestLapS - f.LeaderBestS
	}
	txt := fmt.Sprintf(
		"Сессия %s · карт %s · %d кругов · PB %.3f · медиана %.3f (разброс %.3f сек)",
		valOr(f.TrackName, "—"), valOr(f.Kart, "—"), f.LapCount, f.BestLapS, f.MedianLapS, spread,
	)
	if gap > 0 {
		txt += fmt.Sprintf(" · отставание от лидера трека %.3f сек", gap)
	}
	txt += ". " + localAdvice(spread, gap, len(f.Slips), f.Evidence)
	return &Result{Model: "local", Verdict: txt}
}

func valOr(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

func localAdvice(spread, leaderGap float64, slips int, ev []Evidence) string {
	// Приоритет claim-ов из v3 pipeline Evidence если они есть.
	for _, e := range ev {
		switch e.Claim {
		case "inconsistent_pace":
			return "Нестабильный темп — обрати внимание на точку торможения и траекторию выхода."
		case "best_not_repeatable":
			return "Лучший круг — одиночный флэш. Фокус: воспроизводимость темпа во всей гонке."
		case "trailing_leader":
			return "Отрыв от лидера есть, но PB близок — работай над средним темпом, не над одним кругом."
		}
	}
	switch {
	case spread > 0.4:
		return "Нестабильный темп — обрати внимание на точку торможения и траекторию выхода."
	case slips > 3:
		return "Многократные срывы задней — попробуй мягче на газ при выходе из поворотов."
	case leaderGap > 0.3:
		return "Разрыв с лидером заметен — одного круга мало, нужна стабильность."
	default:
		return "Хорошая стабильность. Пробуй найти ещё 0.1-0.2 на самом сложном повороте."
	}
}

func buildPrompt(f Facts) string {
	b, _ := json.MarshalIndent(f, "", "  ")
	return "Данные сессии:\n" + string(b) + "\n\nДай короткий разбор."
}

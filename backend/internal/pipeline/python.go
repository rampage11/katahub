package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"
)

// RenderAnalysis вызывает Python-воркер katahub_render.py на выгруженной
// сессии (zip). Воркер делает полный pipeline: ориентация, детекция кругов,
// trajectory, фазы (torm/разг/lift/cruise), slip events, сектора. Результат —
// json.RawMessage, пригодный для прямого сохранения в verdicts.facts_json
// и отрисовки SVG на фронте.
//
// На локальных тестах 480 сек Watch IMU → ~2 сек Python processing. 100 МБ
// RAM. Запускаем с timeout 60 сек на всякий случай.
func RenderAnalysis(ctx context.Context, zipPath string) (json.RawMessage, error) {
	script := os.Getenv("PYTHON_ANALYSER_PATH")
	if script == "" {
		script = "/app/analysis/katahub_render.py"
	}
	if _, err := os.Stat(script); err != nil {
		return nil, fmt.Errorf("analyser script not found: %w", err)
	}

	runCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(runCtx, "python3", script, "--zip", zipPath)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

	out, err := cmd.Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return nil, fmt.Errorf("python render: exit=%d stderr=%s", ee.ExitCode(), string(ee.Stderr))
		}
		return nil, fmt.Errorf("python render: %w", err)
	}
	if len(out) == 0 {
		return nil, errors.New("python render: empty output")
	}
	// Валидируем что stdout — валидный JSON (воркер иногда может плюнуть лог
	// в stdout по ошибке; хотим упасть громко если так).
	var probe map[string]any
	if err := json.Unmarshal(out, &probe); err != nil {
		return nil, fmt.Errorf("python render: output is not valid JSON: %w (first 200 bytes: %q)", err, firstN(out, 200))
	}
	log.Printf("pipeline render: %d bytes JSON produced", len(out))
	return json.RawMessage(out), nil
}

func firstN(b []byte, n int) string {
	if len(b) < n {
		return string(b)
	}
	return string(b[:n])
}

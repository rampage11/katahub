package pipeline

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

// ErrAlignGateFailed возвращается когда DTW align quality между двумя
// кругами превышает порог 0.5 rad/s rms. Это признак что круг кривой
// (сход, пит, чужая телеметрия), и compare для него бессмысленен.
// Backend должен ответить 422, dashboard — показать «круг не прошёл гейт».
var ErrAlignGateFailed = errors.New("align gate failed: DTW residual too high")

// BuildCompare генерирует HTML-страницу сравнения двух кругов на базе
// laps_compare.py. Результат пишется атомарно в cachePath (сначала во
// временный файл, потом rename). Параллельные вызовы с одинаковыми
// аргументами де-дуплицируются через in-process lock — Python-скрипт
// тяжёлый (CPU-bound ~2-5 сек), повторный запуск бессмыслен.
//
// Если cachePath уже существует и свежий — функция всё равно его
// пересобирает только при force=true. Иначе возвращает nil сразу.
func BuildCompare(ctx context.Context, zipPath, cachePath string, bestLap, otherLap int, force bool) error {
	if _, err := os.Stat(zipPath); err != nil {
		return fmt.Errorf("zip not found: %w", err)
	}
	if bestLap <= 0 || otherLap <= 0 || bestLap == otherLap {
		return fmt.Errorf("invalid lap pair: best=%d other=%d", bestLap, otherLap)
	}

	if !force {
		if info, err := os.Stat(cachePath); err == nil && info.Size() > 1024 {
			return nil
		}
	}

	lock := compareLockFor(cachePath)
	lock.Lock()
	defer lock.Unlock()
	// Второй вызов мог успеть пока мы ждали.
	if !force {
		if info, err := os.Stat(cachePath); err == nil && info.Size() > 1024 {
			return nil
		}
	}

	script := os.Getenv("PYTHON_COMPARE_PATH")
	if script == "" {
		script = "/app/analysis/laps_compare.py"
	}
	if _, err := os.Stat(script); err != nil {
		return fmt.Errorf("compare script not found: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		return fmt.Errorf("mkdir cache dir: %w", err)
	}

	tmp, err := os.CreateTemp(filepath.Dir(cachePath), ".compare-*.html")
	if err != nil {
		return fmt.Errorf("tempfile: %w", err)
	}
	tmpPath := tmp.Name()
	_ = tmp.Close()
	defer func() {
		if _, err := os.Stat(tmpPath); err == nil {
			_ = os.Remove(tmpPath)
		}
	}()

	runCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(runCtx, "python3", script,
		"--zip", zipPath,
		"--best", strconv.Itoa(bestLap),
		"--other", strconv.Itoa(otherLap),
		"--out", tmpPath,
	)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

	out, err := cmd.CombinedOutput()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			if ee.ExitCode() == 4 {
				return ErrAlignGateFailed
			}
			return fmt.Errorf("compare: exit=%d output=%s", ee.ExitCode(), string(out))
		}
		return fmt.Errorf("compare: %w (output=%s)", err, string(out))
	}

	info, err := os.Stat(tmpPath)
	if err != nil || info.Size() < 1024 {
		return fmt.Errorf("compare: produced empty/short output (%d bytes)", info.Size())
	}
	if err := os.Rename(tmpPath, cachePath); err != nil {
		return fmt.Errorf("rename to cache: %w", err)
	}
	return nil
}

// Lock registry: один mutex на один cache-path. Защищает от параллельных
// запусков тяжёлого Python subprocess на одной и той же паре кругов.
var (
	compareLocks   = make(map[string]*sync.Mutex)
	compareLocksMu sync.Mutex
)

func compareLockFor(path string) *sync.Mutex {
	compareLocksMu.Lock()
	defer compareLocksMu.Unlock()
	if m, ok := compareLocks[path]; ok {
		return m
	}
	m := &sync.Mutex{}
	compareLocks[path] = m
	return m
}

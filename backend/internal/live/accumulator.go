package live

import (
	"sync"

	"github.com/google/uuid"
)

// Accumulator — 120-секундный ring-буфер raw IMU-сэмплов на сессию.
// Используется live-режимом Wave 2: Watch шлёт 1-секундные батчи, бэкенд
// копит их в буфер, на timing.lap событие вырезает [t_start, t_end] и
// пишет laps/{N}.bin.
//
// Еvict oldest когда буфер >120s. Параллельная запись из WS-горутины,
// параллельное чтение из compare-горутины — защищено sync.Mutex.
// Операции O(1) амортизированно за счёт []imuSample append (переаллокация
// 2x только на старте).

const accumulatorWindowNs int64 = 120_000_000_000 // 120 секунд

type Accumulator struct {
	mu      sync.Mutex
	samples []imuSample // отсортированы по TsNs по возрастанию (producer-order)
}

func NewAccumulator() *Accumulator {
	return &Accumulator{samples: make([]imuSample, 0, 100*120+100)}
}

// AddSample добавляет один сэмпл и выкидывает старые (>120с от хвоста).
// Thread-safe. Вызывается из WS-reader на каждый декодированный сэмпл.
func (a *Accumulator) AddSample(s imuSample) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.samples = append(a.samples, s)
	// Эвикт по времени: сдвигаем окно к хвосту.
	cutoff := int64(s.TsNs) - accumulatorWindowNs
	if cutoff <= 0 {
		return
	}
	i := 0
	for ; i < len(a.samples); i++ {
		if int64(a.samples[i].TsNs) >= cutoff {
			break
		}
	}
	if i > 0 {
		// Копируем хвост в начало. Простой подход — пересобрать slice;
		// amortized O(1) если делать не слишком часто.
		copy(a.samples, a.samples[i:])
		a.samples = a.samples[:len(a.samples)-i]
	}
}

// AddBatch добавляет много сэмплов за один захват lock.
func (a *Accumulator) AddBatch(batch []imuSample) {
	if len(batch) == 0 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.samples = append(a.samples, batch...)
	last := batch[len(batch)-1]
	cutoff := int64(last.TsNs) - accumulatorWindowNs
	if cutoff <= 0 {
		return
	}
	i := 0
	for ; i < len(a.samples); i++ {
		if int64(a.samples[i].TsNs) >= cutoff {
			break
		}
	}
	if i > 0 {
		copy(a.samples, a.samples[i:])
		a.samples = a.samples[:len(a.samples)-i]
	}
}

// Slice возвращает копию сэмплов в полуоткрытом интервале [startNs, endNs).
// Полная копия (не view на внутренний slice) чтобы вызывающий код мог
// работать без удержания lock-а.
func (a *Accumulator) Slice(startNs, endNs int64) []imuSample {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]imuSample, 0, 128)
	for _, s := range a.samples {
		ts := int64(s.TsNs)
		if ts < startNs {
			continue
		}
		if ts >= endNs {
			break
		}
		out = append(out, s)
	}
	return out
}

// Snapshot возвращает полный буфер (копию). Для тестов/диагностики.
func (a *Accumulator) Snapshot() []imuSample {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]imuSample, len(a.samples))
	copy(out, a.samples)
	return out
}

// Count — число сэмплов в буфере.
func (a *Accumulator) Count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.samples)
}

// --- registry ---

// AccumulatorRegistry — key=sessionID, value=*Accumulator. Создаётся лениво
// на первый WS-коннект. Жизненный цикл: пока бэкенд жив; при long-running
// гонке на одной сессии буфер держит максимум 120с × 100Hz × 88 байт ≈ 1 МБ.
type AccumulatorRegistry struct {
	mu    sync.RWMutex
	items map[uuid.UUID]*Accumulator
}

func NewAccumulatorRegistry() *AccumulatorRegistry {
	return &AccumulatorRegistry{items: make(map[uuid.UUID]*Accumulator)}
}

func (r *AccumulatorRegistry) GetOrCreate(id uuid.UUID) *Accumulator {
	r.mu.RLock()
	a, ok := r.items[id]
	r.mu.RUnlock()
	if ok {
		return a
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if a, ok := r.items[id]; ok {
		return a
	}
	a = NewAccumulator()
	r.items[id] = a
	return a
}

func (r *AccumulatorRegistry) Get(id uuid.UUID) *Accumulator {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.items[id]
}

// Drop выкидывает буфер (например, после финализации сессии). Не обязателен
// для корректности, но полезен для освобождения памяти.
func (r *AccumulatorRegistry) Drop(id uuid.UUID) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.items, id)
}

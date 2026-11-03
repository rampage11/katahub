package live

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Bus is a fan-out hub: a single publisher (iPhone WebSocket) and many
// subscribers (admin dashboards). Buffered per subscriber; slow consumers
// get dropped frames rather than back-pressuring the producer.
type Bus struct {
	mu   sync.RWMutex
	subs map[uuid.UUID]map[chan Event]struct{}
}

type Event struct {
	SessionID uuid.UUID       `json:"sessionId"`
	WallNs    int64           `json:"wallNs"`
	Kind      string          `json:"kind"`
	Payload   json.RawMessage `json:"payload"`
	ServerTS  int64           `json:"serverTs"`
}

func NewBus() *Bus {
	return &Bus{subs: make(map[uuid.UUID]map[chan Event]struct{})}
}

func (b *Bus) Subscribe(sess uuid.UUID) (<-chan Event, func()) {
	ch := make(chan Event, 128)
	b.mu.Lock()
	if _, ok := b.subs[sess]; !ok {
		b.subs[sess] = make(map[chan Event]struct{})
	}
	b.subs[sess][ch] = struct{}{}
	b.mu.Unlock()
	cancel := func() {
		b.mu.Lock()
		if m, ok := b.subs[sess]; ok {
			delete(m, ch)
			if len(m) == 0 {
				delete(b.subs, sess)
			}
		}
		b.mu.Unlock()
		close(ch)
	}
	return ch, cancel
}

func (b *Bus) Publish(ev Event) {
	ev.ServerTS = time.Now().UnixNano()
	b.mu.RLock()
	subs := b.subs[ev.SessionID]
	// Snapshot the map to avoid holding the lock during sends.
	targets := make([]chan Event, 0, len(subs))
	for ch := range subs {
		targets = append(targets, ch)
	}
	b.mu.RUnlock()
	for _, ch := range targets {
		select {
		case ch <- ev:
		default:
			// drop — slow subscriber
		}
	}
}

func (b *Bus) Active(sess uuid.UUID) int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.subs[sess])
}

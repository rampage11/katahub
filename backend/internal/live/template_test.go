package live

import (
	"os"
	"testing"
)

// TestLoadTemplateNPZ требует /tmp/miks.npz (кладёт build_track_template.py).
// Если файла нет — skip, чтобы go test ./... не падал на свежей машине.
func TestLoadTemplateNPZ(t *testing.T) {
	path := "/tmp/miks.npz"
	if _, err := os.Stat(path); err != nil {
		t.Skip("no miks.npz at /tmp, skip")
	}
	tpl, err := LoadTemplateNPZ(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tpl.Fs != 50.0 {
		t.Fatalf("fs=%v want 50", tpl.Fs)
	}
	if len(tpl.Wz) != 1643 {
		t.Fatalf("len(wz)=%d want 1643", len(tpl.Wz))
	}
	if len(tpl.Wz) != len(tpl.Ah) || len(tpl.Wz) != len(tpl.V) {
		t.Fatalf("inconsistent lengths wz=%d ah=%d v=%d", len(tpl.Wz), len(tpl.Ah), len(tpl.V))
	}
	if tpl.Meta == "" {
		t.Fatalf("meta empty, want JSON string")
	}
	// Проверяем TailHeadWindow: 5+5с @ 50Гц = 500 образцов.
	win := tpl.TailHeadWindow(5, 5)
	if len(win.Wz) != 500 {
		t.Fatalf("tail+head window len=%d want 500", len(win.Wz))
	}
	if win.TailN != 250 {
		t.Fatalf("tailN=%d want 250", win.TailN)
	}
}

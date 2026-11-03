package live

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
)

// Template хранит эталонный круг для текущей трассы, построенный заранее
// скриптом build_track_template.py. Содержит wz/|a|/v/x/y @ 50 Гц.
// Используется в Wave 2 live-режиме для:
//   1. RefineBoundary — локальный поиск границы круга (±2 с от wall-time)
//      по корреляции wz с "хвост+голова" шаблона.
//   2. NCC-гейт — отсечение явно левых окон (пит, сход).
//
// Шаблон строится один раз и грузится при старте процесса. При смене
// трассы бэкенд перезагружается с новым /app/tracks/<name>.npz.
type Template struct {
	Fs        float64
	Wz        []float32
	Ah        []float32
	V         []float32
	X         []float32
	Y         []float32
	Meta      string
	LapTimeS  float64
	Loaded    bool
}

// TemplateWindow — "хвост+голова" шаблона: последние tailS секунд + первые
// headS секунд, подряд. Нужно для скользящей корреляции сигналов которые
// пересекают линию старт/финиш (конкатенируются вокруг S/F).
type TemplateWindow struct {
	Wz    []float64
	Fs    float64
	TailN int // сколько образцов из хвоста (голова = len(Wz) - TailN)
}

var (
	templateMu  sync.RWMutex
	templateCur *Template
)

// SetGlobalTemplate регистрирует загруженный шаблон как текущий.
func SetGlobalTemplate(t *Template) {
	templateMu.Lock()
	defer templateMu.Unlock()
	templateCur = t
}

// GlobalTemplate возвращает текущий шаблон (может быть nil, если не загружен).
func GlobalTemplate() *Template {
	templateMu.RLock()
	defer templateMu.RUnlock()
	return templateCur
}

// LoadTemplateNPZ парсит numpy .npz (zip-of-npy) с полями fs, wz, ah, v,
// x, y, meta. Все числовые массивы — float32 LE. meta — object array
// с одной JSON-строкой.
//
// Делаем свой парсер вместо вызова python чтобы не тащить subprocess
// в hot path при старте.
func LoadTemplateNPZ(path string) (*Template, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open npz: %w", err)
	}
	defer zr.Close()

	t := &Template{}
	for _, f := range zr.File {
		name := strings.TrimSuffix(f.Name, ".npy")
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, err
		}
		switch name {
		case "fs":
			// fs может быть f8 (np.array([50.0]) default) или f4.
			if v, err := parseNpyFloatScalar(data); err != nil {
				return nil, fmt.Errorf("fs: %w", err)
			} else {
				t.Fs = v
			}
		case "wz":
			arr, err := parseNpyFloat32(data)
			if err != nil {
				return nil, fmt.Errorf("wz: %w", err)
			}
			t.Wz = arr
		case "ah":
			arr, err := parseNpyFloat32(data)
			if err != nil {
				return nil, fmt.Errorf("ah: %w", err)
			}
			t.Ah = arr
		case "v":
			arr, err := parseNpyFloat32(data)
			if err != nil {
				return nil, fmt.Errorf("v: %w", err)
			}
			t.V = arr
		case "x":
			arr, err := parseNpyFloat32(data)
			if err != nil {
				return nil, fmt.Errorf("x: %w", err)
			}
			t.X = arr
		case "y":
			arr, err := parseNpyFloat32(data)
			if err != nil {
				return nil, fmt.Errorf("y: %w", err)
			}
			t.Y = arr
		case "meta":
			meta, err := parseNpyObjectJSON(data)
			if err != nil {
				// meta не критичный, продолжаем
				t.Meta = ""
			} else {
				t.Meta = meta
			}
		}
	}
	if t.Fs <= 0 || len(t.Wz) == 0 {
		return nil, errors.New("npz missing required fs/wz")
	}
	t.Loaded = true
	return t, nil
}

// TailHeadWindow возвращает "хвост+голова" из текущего шаблона: последние
// tailS секунд + первые headS секунд, конкатенированные. Это окно для NCC
// при RefineBoundary — реальная граница круга лежит внутри него.
func (t *Template) TailHeadWindow(tailS, headS float64) TemplateWindow {
	if t == nil || t.Fs <= 0 || len(t.Wz) == 0 {
		return TemplateWindow{}
	}
	tailN := int(tailS * t.Fs)
	headN := int(headS * t.Fs)
	if tailN > len(t.Wz) {
		tailN = len(t.Wz)
	}
	if headN > len(t.Wz) {
		headN = len(t.Wz)
	}
	out := make([]float64, 0, tailN+headN)
	for i := len(t.Wz) - tailN; i < len(t.Wz); i++ {
		out = append(out, float64(t.Wz[i]))
	}
	for i := 0; i < headN; i++ {
		out = append(out, float64(t.Wz[i]))
	}
	return TemplateWindow{Wz: out, Fs: t.Fs, TailN: tailN}
}

// LapDurationS — длительность эталонного круга (len(Wz) / Fs).
func (t *Template) LapDurationS() float64 {
	if t == nil || t.Fs <= 0 {
		return 0
	}
	return float64(len(t.Wz)) / t.Fs
}

// --- парсер .npy ---

// Numpy .npy format: 6-byte magic "\x93NUMPY" + 2-byte version
// (0x01 0x00) + 2-byte LE header length + ASCII header dict + data.
// Для v2 header length — 4 байта LE. Нам нужны v1 (так пишет build_track_template).

type npyHeader struct {
	Descr        string
	FortranOrder bool
	Shape        []int
	DataOffset   int
}

func parseNpyHeader(data []byte) (npyHeader, error) {
	if len(data) < 10 || string(data[0:6]) != "\x93NUMPY" {
		return npyHeader{}, errors.New("not npy")
	}
	major := data[6]
	// minor := data[7]
	var headerLen, headerStart int
	if major == 1 {
		headerLen = int(binary.LittleEndian.Uint16(data[8:10]))
		headerStart = 10
	} else if major == 2 || major == 3 {
		headerLen = int(binary.LittleEndian.Uint32(data[8:12]))
		headerStart = 12
	} else {
		return npyHeader{}, fmt.Errorf("npy version %d unsupported", major)
	}
	dataOffset := headerStart + headerLen
	if dataOffset > len(data) {
		return npyHeader{}, errors.New("npy header overrun")
	}
	hdr := string(data[headerStart:dataOffset])

	descr := extractNpyKV(hdr, "'descr'")
	descr = strings.Trim(descr, "' ")
	fortran := strings.Contains(extractNpyKV(hdr, "'fortran_order'"), "True")
	shapeRaw := extractNpyKV(hdr, "'shape'")
	// shapeRaw like "(1643,)" or "(10, 2)"
	shapeRaw = strings.TrimSpace(shapeRaw)
	shapeRaw = strings.Trim(shapeRaw, "()")
	shape := []int{}
	for _, p := range strings.Split(shapeRaw, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			return npyHeader{}, fmt.Errorf("shape token %q: %w", p, err)
		}
		shape = append(shape, n)
	}
	return npyHeader{
		Descr:        descr,
		FortranOrder: fortran,
		Shape:        shape,
		DataOffset:   dataOffset,
	}, nil
}

// extractNpyKV — грубый ad-hoc парсер ASCII-дикта из .npy header.
// Структура всегда повторяемая: "{'descr': '<f4', 'fortran_order': False, 'shape': (1643,), }",
// нам достаточно вытащить значение по ключу-подстроке до следующей запятой-на-верхнем-уровне.
func extractNpyKV(hdr, key string) string {
	i := strings.Index(hdr, key)
	if i < 0 {
		return ""
	}
	rest := hdr[i+len(key):]
	// skip until ':'
	colon := strings.Index(rest, ":")
	if colon < 0 {
		return ""
	}
	rest = rest[colon+1:]
	// значение заканчивается на запятой на верхнем уровне (т.е. снаружи
	// любых скобок). Простой счётчик скобок справляется.
	depth := 0
	end := len(rest)
	for i := 0; i < len(rest); i++ {
		c := rest[i]
		switch c {
		case '(', '[', '{':
			depth++
		case ')', ']', '}':
			depth--
			if depth < 0 {
				end = i
				goto done
			}
		case ',':
			if depth == 0 {
				end = i
				goto done
			}
		}
	}
done:
	return strings.TrimSpace(rest[:end])
}

// parseNpyFloatScalar читает первый элемент как float64, независимо от того
// `<f4` или `<f8` dtype.
func parseNpyFloatScalar(data []byte) (float64, error) {
	hdr, err := parseNpyHeader(data)
	if err != nil {
		return 0, err
	}
	raw := data[hdr.DataOffset:]
	switch hdr.Descr {
	case "<f4":
		if len(raw) < 4 {
			return 0, errors.New("short f4 scalar")
		}
		var v float32
		r := bytes.NewReader(raw[:4])
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return 0, err
		}
		return float64(v), nil
	case "<f8":
		if len(raw) < 8 {
			return 0, errors.New("short f8 scalar")
		}
		var v float64
		r := bytes.NewReader(raw[:8])
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return 0, err
		}
		return v, nil
	default:
		return 0, fmt.Errorf("fs dtype %q unsupported", hdr.Descr)
	}
}

func parseNpyFloat32(data []byte) ([]float32, error) {
	hdr, err := parseNpyHeader(data)
	if err != nil {
		return nil, err
	}
	if hdr.Descr != "<f4" {
		return nil, fmt.Errorf("expected <f4 got %q", hdr.Descr)
	}
	n := 1
	for _, s := range hdr.Shape {
		n *= s
	}
	raw := data[hdr.DataOffset:]
	if len(raw) < n*4 {
		return nil, fmt.Errorf("short data: have %d need %d", len(raw), n*4)
	}
	out := make([]float32, n)
	r := bytes.NewReader(raw)
	if err := binary.Read(r, binary.LittleEndian, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// parseNpyObjectJSON вытаскивает JSON-строку из object-массива .npy (shape (1,),
// descr "|O"). Numpy сериализует object-dtype через pickle, но наш частный
// случай — одна строка. Пикл-протокол для строки состоит из:
//   \x80\x04 ...  BINUNICODE: short string: 0x8C <len:1> <bytes> или
//   длинная: 0x8D <len:4> <bytes>.
// Проще найти первую { в payload и до соответствующей } извлечь как JSON.
func parseNpyObjectJSON(data []byte) (string, error) {
	hdr, err := parseNpyHeader(data)
	if err != nil {
		return "", err
	}
	raw := data[hdr.DataOffset:]
	// грубое: ищем первую '{' и считаем скобки до парной '}'.
	start := bytes.IndexByte(raw, '{')
	if start < 0 {
		return "", errors.New("no JSON found")
	}
	depth := 0
	for i := start; i < len(raw); i++ {
		switch raw[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return string(raw[start : i+1]), nil
			}
		}
	}
	return "", errors.New("unbalanced JSON")
}

// MustStatTemplate — утилита для main: вернуть размер в байтах или -1
// если файла нет. Используется для диагностики при старте.
func MustStatTemplate(path string) int64 {
	st, err := os.Stat(path)
	if err != nil {
		return -1
	}
	return st.Size()
}

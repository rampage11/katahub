package live

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

// Wave 2 wire-формат батчей Watch → katahub.
//
// Header (8 байт, little-endian):
//   [0..2) magic = 0x494D ('I','M')
//   [2..3) version = 1
//   [3..4) reserved = 0
//   [4..8) count: uint32
// Затем count × 88 байт = IMUSampleBinary (как в watch_imu.bin):
//   ts_ns u64, ax/ay/az/gx/gy/gz/qw/qx/qy/qz f64 (всё LE).
//
// Приходит по /live/{id}/watch-imu как binary frame; iPhone открывает этот
// WS только когда получает первый батч с часов, не раньше.

const (
	batchMagic      uint16 = 0x494D
	batchHeaderSize        = 8
)

// DecodeWatchImuBatch разбирает один binary-кадр. Возвращает slice сэмплов
// или ошибку (битая magic/version/размер).
func DecodeWatchImuBatch(frame []byte) ([]imuSample, error) {
	if len(frame) < batchHeaderSize {
		return nil, errors.New("short frame")
	}
	magic := binary.LittleEndian.Uint16(frame[0:2])
	if magic != batchMagic {
		return nil, fmt.Errorf("bad magic 0x%04x", magic)
	}
	version := frame[2]
	if version != 1 {
		return nil, fmt.Errorf("unsupported version %d", version)
	}
	count := binary.LittleEndian.Uint32(frame[4:8])
	payload := frame[batchHeaderSize:]
	expectedSize := int(count) * imuSampleSize
	if len(payload) < expectedSize {
		return nil, fmt.Errorf("truncated: have %d need %d", len(payload), expectedSize)
	}
	out := make([]imuSample, count)
	r := bytes.NewReader(payload[:expectedSize])
	for i := range out {
		if err := binary.Read(r, binary.LittleEndian, &out[i]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

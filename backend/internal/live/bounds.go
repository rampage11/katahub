package live

import (
	"math"
	"time"
)

// RefineBoundary ищет точную границу круга в окне ±searchS секунд вокруг
// заявленного wall-time, используя нормализованную кросс-корреляцию
// (NCC) сигнала wz против "хвост+голова" шаблона.
//
// Логика:
//   * Заявленный wall-time (`finishWall`) — отсечка в timing.json. Точность
//     RaceMann ±1 с (сетевые задержки + клиентский клок-дрейф).
//   * Берём участок accumulator-а [finishWall - lapS - searchS, finishWall + searchS]
//     и ресемплим в сетку шаблона (fs=50Гц).
//   * Шаблонное окно — последние tailS секунд предыдущего круга + первые headS
//     секунд следующего = ~10с. Это сигнатура прохождения S/F — на реальной
//     трассе характерный паттерн (разгон после последнего поворота + торможение
//     перед первым).
//   * Скользящая NCC: двигаем шаблон по accumulator-окну, ищем position с
//     максимальным скаляром → это точка где "хвост" заканчивается, т.е. S/F.
//
// Возвращает startNs/endNs — границы круга в accumulator-коордах, nccScore —
// качество выравнивания. Если accumulator пуст или шаблона нет, возвращает
// (finishWall-lap, finishWall, 0).
func RefineBoundary(
	finishWall time.Time,
	lapMs int,
	acc *Accumulator,
	tpl *Template,
	searchS float64,
) (startNs, endNs int64, ncc float64) {
	lapS := float64(lapMs) / 1000.0
	finishNs := finishWall.UnixNano()
	rawStart := finishNs - int64(lapS*1e9)
	if tpl == nil || !tpl.Loaded || acc == nil {
		return rawStart, finishNs, 0
	}

	// Нам нужен "хвост" tail=5с и "голова" head=5с для шаблона. Чтобы
	// NCC-окно целиком лежало в accumulator, набираем [finishWall-5c-search,
	// finishWall+5c+search].
	tailS, headS := 5.0, 5.0
	win := tpl.TailHeadWindow(tailS, headS)
	if len(win.Wz) < 10 {
		return rawStart, finishNs, 0
	}
	windowFs := win.Fs
	// NCC-окно acc = [finishNs - int64((tailS+searchS)*1e9), finishNs + int64((headS+searchS)*1e9)]
	accStart := finishNs - int64((tailS+searchS)*1e9)
	accEnd := finishNs + int64((headS+searchS)*1e9)
	raw := acc.Slice(accStart, accEnd)
	if len(raw) < 10 {
		return rawStart, finishNs, 0
	}

	// Ресемпл wz raw в сетку шаблона fs. Простой nearest-neighbor —
	// template fs=50Гц, raw fs=100Гц, достаточно точно.
	ngrid := int(float64(accEnd-accStart) * windowFs / 1e9)
	if ngrid < len(win.Wz)+2 {
		return rawStart, finishNs, 0
	}
	rawWz := make([]float64, ngrid)
	dtNs := int64(1e9 / windowFs) // 20мс при 50Гц
	// Указатель на raw который продвигается монотонно.
	j := 0
	for i := 0; i < ngrid; i++ {
		gridNs := accStart + int64(i)*dtNs
		for j+1 < len(raw) && int64(raw[j+1].TsNs) <= gridNs {
			j++
		}
		rawWz[i] = raw[j].Gz
	}

	// Скользящая NCC: шаблон длины M, raw длины N; максимум по 0..N-M.
	M := len(win.Wz)
	N := len(rawWz)
	if N < M+2 {
		return rawStart, finishNs, 0
	}
	// Предсчёт норм template.
	var tplMean, tplVar float64
	for _, v := range win.Wz {
		tplMean += v
	}
	tplMean /= float64(M)
	for _, v := range win.Wz {
		d := v - tplMean
		tplVar += d * d
	}
	tplNorm := math.Sqrt(tplVar)
	if tplNorm < 1e-9 {
		return rawStart, finishNs, 0
	}

	bestNCC := -2.0
	bestLag := 0
	for lag := 0; lag <= N-M; lag++ {
		// local mean/var
		var m, s2 float64
		for i := 0; i < M; i++ {
			m += rawWz[lag+i]
		}
		m /= float64(M)
		var dot float64
		for i := 0; i < M; i++ {
			dw := rawWz[lag+i] - m
			dt := win.Wz[i] - tplMean
			dot += dw * dt
			s2 += dw * dw
		}
		localNorm := math.Sqrt(s2)
		if localNorm < 1e-9 {
			continue
		}
		c := dot / (localNorm * tplNorm)
		if c > bestNCC {
			bestNCC = c
			bestLag = lag
		}
	}

	// bestLag — смещение в сетке; "хвост" заканчивается на (bestLag + tailN).
	// То есть мы нашли конкретный grid-индекс финиша: fin = bestLag + tailN
	finGridNs := accStart + int64(bestLag+win.TailN)*dtNs
	// Начало круга = fin - lapS. lapS берётся из timing (авторитет); мы
	// рефайним только позицию финиша, не продолжительность.
	refinedEndNs := finGridNs
	refinedStartNs := refinedEndNs - int64(lapS*1e9)
	return refinedStartNs, refinedEndNs, bestNCC
}

// NccVsTemplate — грубая оценка качества круга: NCC полного wz лапа против
// полного шаблона после resample. Используется как save-гейт (≥0.6 → сохраняем).
func NccVsTemplate(lap []imuSample, tpl *Template) float64 {
	if tpl == nil || !tpl.Loaded || len(lap) < 10 || len(tpl.Wz) < 10 {
		return 0
	}
	// Ресэмплим lap wz в сетку шаблона: равномерно по длительности.
	lapDurNs := int64(lap[len(lap)-1].TsNs - lap[0].TsNs)
	if lapDurNs <= 0 {
		return 0
	}
	n := len(tpl.Wz)
	// time → index map через линейный поиск; lap короткий (<1500 образцов),
	// n=1600, итого ≤2.4М сравнений, за ms.
	lapWz := make([]float64, n)
	j := 0
	for i := 0; i < n; i++ {
		frac := float64(i) / float64(n-1)
		tNs := int64(lap[0].TsNs) + int64(frac*float64(lapDurNs))
		for j+1 < len(lap) && int64(lap[j+1].TsNs) <= tNs {
			j++
		}
		lapWz[i] = lap[j].Gz
	}
	// Pearson корреляция = NCC для одинаковой длины.
	var mA, mB float64
	for i := 0; i < n; i++ {
		mA += lapWz[i]
		mB += float64(tpl.Wz[i])
	}
	mA /= float64(n)
	mB /= float64(n)
	var sAA, sBB, sAB float64
	for i := 0; i < n; i++ {
		dA := lapWz[i] - mA
		dB := float64(tpl.Wz[i]) - mB
		sAA += dA * dA
		sBB += dB * dB
		sAB += dA * dB
	}
	denom := math.Sqrt(sAA * sBB)
	if denom < 1e-9 {
		return 0
	}
	return sAB / denom
}

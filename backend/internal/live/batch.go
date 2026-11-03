package live

import (
	"fmt"
	"math"
)

// Batch-процессор для replay: все данные есть сразу, θ ищется один раз
// по всему окну, aLong сглаживается двухпроходно. Голос за иллюзию
// каузальности мы в live-режиме отрабатываем отдельным ThetaEstimator
// в iOS, здесь — честный offline.

// gaussianSmooth применяет симметричное гауссово сглаживание σ = sigma
// (в сэмплах). Эквивалентно scipy.ndimage.gaussian_filter1d с mode='reflect'.
func gaussianSmooth(x []float64, sigma float64) []float64 {
	if sigma <= 0 || len(x) == 0 {
		out := make([]float64, len(x))
		copy(out, x)
		return out
	}
	radius := int(math.Ceil(sigma * 4))
	if radius < 1 {
		radius = 1
	}
	// kernel
	kern := make([]float64, 2*radius+1)
	s := 0.0
	for i := -radius; i <= radius; i++ {
		kern[i+radius] = math.Exp(-float64(i*i) / (2 * sigma * sigma))
		s += kern[i+radius]
	}
	for i := range kern {
		kern[i] /= s
	}
	n := len(x)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		var acc float64
		for k := -radius; k <= radius; k++ {
			j := i + k
			if j < 0 {
				j = -j
			} else if j >= n {
				j = 2*n - 2 - j
				if j < 0 {
					j = 0
				}
			}
			acc += x[j] * kern[k+radius]
		}
		out[i] = acc
	}
	return out
}

// pearsonCorr — обычная корреляция Пирсона двух равных массивов.
func pearsonCorr(a, b []float64) float64 {
	n := len(a)
	if n == 0 || n != len(b) {
		return 0
	}
	var sx, sy, sxy, sxx, syy float64
	for i := 0; i < n; i++ {
		sx += a[i]
		sy += b[i]
		sxy += a[i] * b[i]
		sxx += a[i] * a[i]
		syy += b[i] * b[i]
	}
	fn := float64(n)
	cov := sxy/fn - (sx/fn)*(sy/fn)
	vx := sxx/fn - (sx/fn)*(sx/fn)
	vy := syy/fn - (sy/fn)*(sy/fn)
	denom := math.Sqrt(vx * vy)
	if denom < 1e-9 {
		return 0
	}
	return cov / denom
}

// BatchProcess делает весь pipeline offline по массиву сэмплов:
// 1) проверка знака кватерниона (|∑wz·dt|/2π ∈ [0.9, 1.1] × число кругов);
// 2) ротация body→world, unwrapped ψ;
// 3) θ = argmax corr(a_lat(θ), lowpass(wz)) по всему окну, шаг 1°;
// 4) aLong сглаживается двухпроходно;
// 5) фазы с гистерезисом + dwell классифицируются.
// Возвращает slice phases длины len(samples) и meta (θ в градусах,
// флаг инвертирования кватерниона, корреляция).
type BatchResult struct {
	Phases       []string
	ThetaDeg     float64
	QInverted    bool
	Corr         float64
	ALongSmooth  []float64
	Yaw          []float64 // unwrapped
	Note         string
}

func BatchProcess(samples []imuSample, fs float64) BatchResult {
	n := len(samples)
	ax := make([]float64, n)
	ay := make([]float64, n)
	wz := make([]float64, n)
	yaw := make([]float64, n)

	rotateAll := func(invert bool) {
		var prevY float64
		for i := 0; i < n; i++ {
			s := samples[i]
			qw, qx, qy, qz := s.Qw, s.Qx, s.Qy, s.Qz
			if invert {
				qx, qy, qz = -qx, -qy, -qz
			}
			ax[i], ay[i], _ = rotateBodyToWorld(s.Ax, s.Ay, s.Az, qw, qx, qy, qz)
			_, _, wz[i] = rotateBodyToWorld(s.Gx, s.Gy, s.Gz, qw, qx, qy, qz)
			y := math.Atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
			if i == 0 {
				yaw[i] = y
				prevY = y
			} else {
				d := y - prevY
				if d > math.Pi {
					d -= 2 * math.Pi
				} else if d < -math.Pi {
					d += 2 * math.Pi
				}
				yaw[i] = yaw[i-1] + d
				prevY = y
			}
		}
	}

	// Прогоняем без инверсии, проверяем общее yaw.
	rotateAll(false)
	totalYaw := yaw[n-1] - yaw[0]
	// Хорошая проверка: |totalYaw|/(2π) должно быть ≈ целому числу кругов
	// в допуске 0.9..1.1 × round(.) — иначе кватернион подозрителен.
	laps := math.Round(math.Abs(totalYaw) / (2 * math.Pi))
	if laps < 1 {
		laps = 1
	}
	frac := math.Abs(math.Abs(totalYaw)/(2*math.Pi) - laps)
	inverted := false
	if frac > 0.3 {
		// пробуем инвертировать
		rotateAll(true)
		totalYaw2 := yaw[n-1] - yaw[0]
		laps2 := math.Round(math.Abs(totalYaw2) / (2 * math.Pi))
		if laps2 < 1 {
			laps2 = 1
		}
		frac2 := math.Abs(math.Abs(totalYaw2)/(2*math.Pi) - laps2)
		if frac2 < frac {
			inverted = true
			totalYaw = totalYaw2
			laps = laps2
			frac = frac2
		} else {
			rotateAll(false) // возвращаемся к исходному
		}
	}

	// Шаг 3: θ grid search по всему окну, по сглаженным сигналам
	// (σ=12 сэмплов ≈ 0.12с при 100Гц). Вибрация запястья не должна
	// доминировать в корреляции.
	ax_s := gaussianSmooth(ax, 12)
	ay_s := gaussianSmooth(ay, 12)
	wz_s := gaussianSmooth(wz, 12)
	bestCorr := -2.0
	bestThDeg := 0
	// a_lat(θ) = -ax·sin(ψ+θ) + ay·cos(ψ+θ)
	for deg := 0; deg < 360; deg++ {
		th := float64(deg) * math.Pi / 180.0
		alat := make([]float64, n)
		for i := 0; i < n; i++ {
			c := math.Cos(yaw[i] + th)
			si := math.Sin(yaw[i] + th)
			alat[i] = -ax_s[i]*si + ay_s[i]*c
		}
		c := pearsonCorr(alat, wz_s)
		if c > bestCorr {
			bestCorr = c
			bestThDeg = deg
		}
	}

	// Шаг 4: aLong со сглаживанием σ=8 (≈0.08c по эталону mksvg2.py).
	th := float64(bestThDeg) * math.Pi / 180.0
	computeALong := func(theta float64) []float64 {
		out := make([]float64, n)
		for i := 0; i < n; i++ {
			c := math.Cos(yaw[i] + theta)
			si := math.Sin(yaw[i] + theta)
			out[i] = ax[i]*c + ay[i]*si
		}
		return out
	}
	aLong := computeALong(th)
	aLongSm := gaussianSmooth(aLong, 8)

	// Шаг 4b (per feedback): проверка знака по ФИЗИКЕ. θ и θ+180° дают
	// противоположный знак a_long. Max corr(a_lat, wz) неразличает их, если
	// одновременно wz перевёрнут. Робастный критерий: ПЕРЕД apex тормозят,
	// ПОСЛЕ apex разгоняются → mean(aLong, [apex-1s, apex]) < mean(aLong, [apex, apex+1s]).
	// Голосование по всем поворотам: votes > 0 → знак правильный; votes < 0
	// → θ += 180°, aLong пересчитать.
	wzAbs := make([]float64, n)
	for i := range wz {
		wzAbs[i] = math.Abs(wz_s[i])
	}
	w1s := int(1.5 * fs)
	minDist := int(2.0 * fs) // >=2с между пиками → один пик на корнер
	// Peak detector с минимальной дистанцией 2с и threshold 1.0 рад/с.
	// Внутри каждого окна берём абсолютный максимум — чище чем "любой
	// локальный пик".
	peaks := []int{}
	for i := minDist; i < n-minDist; {
		if wzAbs[i] < 1.0 {
			i++
			continue
		}
		// Нашли кандидата; ищем максимум в окне [i, i+2*minDist]
		end := i + 2*minDist
		if end > n {
			end = n
		}
		maxI := i
		for k := i; k < end; k++ {
			if wzAbs[k] > wzAbs[maxI] {
				maxI = k
			}
		}
		peaks = append(peaks, maxI)
		i = maxI + minDist
	}
	// Голосование: разница mean(after) - mean(before). Если положительна —
	// пилот разгоняется после апекса (правильно). Отрицательна — тормозит
	// (значит знак инвертирован).
	totalDelta := 0.0
	for _, p := range peaks {
		if p-w1s < 0 || p+w1s >= n {
			continue
		}
		before := 0.0
		after := 0.0
		for k := 0; k < w1s; k++ {
			before += aLongSm[p-w1s+k]
			after += aLongSm[p+k]
		}
		totalDelta += (after - before) / float64(w1s)
	}
	signNote := fmt.Sprintf("sign-check: %d peaks, meanΔ=%+.3f g", len(peaks), totalDelta/math.Max(1, float64(len(peaks))))
	if totalDelta < 0 {
		th += math.Pi
		aLong = computeALong(th)
		aLongSm = gaussianSmooth(aLong, 8)
		bestThDeg = (bestThDeg + 180) % 360
		signNote += " → flipped θ by 180°"
	}

	// Шаг 5: классификация с гистерезисом и min-dwell (sample-level).
	classify := func(y float64) string {
		switch {
		case y < -0.25:
			return "brake"
		case y < -0.08:
			return "lift"
		case y > 0.20:
			return "accel"
		default:
			return "cruise"
		}
	}
	phases := make([]string, n)
	phase := "cruise"
	sinceSample := 0
	minDwellSamples := int(0.15 * fs)
	for i := 0; i < n; i++ {
		y := aLongSm[i]
		var next string
		switch phase {
		case "brake":
			if y > -0.18 {
				next = classify(y)
			} else {
				next = "brake"
			}
		case "accel":
			if y < 0.12 {
				next = classify(y)
			} else {
				next = "accel"
			}
		default:
			next = classify(y)
		}
		if next != phase && (i-sinceSample) < minDwellSamples {
			phases[i] = phase
			continue
		}
		if next != phase {
			phase = next
			sinceSample = i
		}
		phases[i] = phase
	}

	return BatchResult{
		Phases:      phases,
		ThetaDeg:    float64(bestThDeg),
		QInverted:   inverted,
		Corr:        bestCorr,
		ALongSmooth: aLongSm,
		Yaw:         yaw,
		Note:        "batch (acausal); " + signNote,
	}
}

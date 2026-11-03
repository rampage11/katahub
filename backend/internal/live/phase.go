package live

import (
	"math"
)

// Rotation body→world через кватернион (xyzw convention), эквивалент
// scipy.spatial.transform.Rotation.from_quat([x,y,z,w]).apply(v).
func rotateBodyToWorld(vx, vy, vz, qw, qx, qy, qz float64) (float64, float64, float64) {
	xx := qx * qx
	yy := qy * qy
	zz := qz * qz
	xy := qx * qy
	xz := qx * qz
	yz := qy * qz
	wx := qw * qx
	wy := qw * qy
	wzq := qw * qz
	rx := (1-2*(yy+zz))*vx + 2*(xy-wzq)*vy + 2*(xz+wy)*vz
	ry := 2*(xy+wzq)*vx + (1-2*(xx+zz))*vy + 2*(yz-wx)*vz
	rz := 2*(xz-wy)*vx + 2*(yz+wx)*vy + (1-2*(xx+yy))*vz
	return rx, ry, rz
}

// quaternionYaw из кватерниона (xyzw), unwrapped на вызывающей стороне.
func quaternionYaw(qw, qx, qy, qz float64) float64 {
	return math.Atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
}

// ThetaEstimator ищет θ, максимизирующий corr(a_lat(θ), wz) в скользящем
// окне (per user feedback). Для телефона на раме θ константа после первого
// круга — EMA сглаживание даст стабильную оценку.
type ThetaEstimator struct {
	bufSize   int
	ax, ay    []float64
	wz, psi   []float64
	idx       int
	filled    bool
	theta     float64
	alphaEMA  float64 // EMA постоянная для сглаживания θ между обновлениями
	lastEst   float64 // wall seconds последнего пересчёта
	warmupSec float64
	startWall float64
}

func NewThetaEstimator(bufSize int, warmupSec float64) *ThetaEstimator {
	return &ThetaEstimator{
		bufSize:   bufSize,
		ax:        make([]float64, bufSize),
		ay:        make([]float64, bufSize),
		wz:        make([]float64, bufSize),
		psi:       make([]float64, bufSize),
		alphaEMA:  0.3,
		warmupSec: warmupSec,
	}
}

func (e *ThetaEstimator) Update(ax, ay, wz, psi, wallSec float64) float64 {
	if e.startWall == 0 {
		e.startWall = wallSec
	}
	e.ax[e.idx] = ax
	e.ay[e.idx] = ay
	e.wz[e.idx] = wz
	e.psi[e.idx] = psi
	e.idx = (e.idx + 1) % e.bufSize
	if e.idx == 0 {
		e.filled = true
	}
	// пересчёт раз в секунду, если буфер хоть частично полон
	if wallSec-e.lastEst < 1.0 {
		return e.theta
	}
	n := e.bufSize
	if !e.filled {
		n = e.idx
	}
	if n < 20 {
		return e.theta
	}
	best := -2.0
	bestTh := e.theta
	for deg := 0; deg < 360; deg += 10 {
		th := float64(deg) * math.Pi / 180.0
		sumX, sumY, sumXY, sumX2, sumY2 := 0.0, 0.0, 0.0, 0.0, 0.0
		for i := 0; i < n; i++ {
			c := math.Cos(e.psi[i] + th)
			s := math.Sin(e.psi[i] + th)
			alat := -e.ax[i]*s + e.ay[i]*c
			w := e.wz[i]
			sumX += alat
			sumY += w
			sumXY += alat * w
			sumX2 += alat * alat
			sumY2 += w * w
		}
		fn := float64(n)
		cov := sumXY/fn - (sumX/fn)*(sumY/fn)
		vx := sumX2/fn - (sumX/fn)*(sumX/fn)
		vy := sumY2/fn - (sumY/fn)*(sumY/fn)
		denom := math.Sqrt(vx * vy)
		if denom < 1e-9 {
			continue
		}
		corr := cov / denom
		if corr > best {
			best = corr
			bestTh = th
		}
	}
	e.theta = e.theta*(1-e.alphaEMA) + bestTh*e.alphaEMA
	e.lastEst = wallSec
	return e.theta
}

func (e *ThetaEstimator) IsCalibrated(wallSec float64) bool {
	return (wallSec - e.startWall) >= e.warmupSec
}

// PhaseFilter: однополюсный НЧ с τ=0.15с + гистерезис + min dwell 150мс.
// Используется на ВСЕХ 100Гц сэмплах; классификация обновляется каждый
// шаг, публикуем каждый 10-й.
type PhaseFilter struct {
	y         float64 // сглаженный a_long в g
	tau       float64 // 0.15с
	prevT     float64 // wall seconds предыдущего step
	phase     string
	since     float64 // когда зафиксировали текущую фазу
	minDwell  float64 // 0.15с
}

func NewPhaseFilter() *PhaseFilter {
	return &PhaseFilter{tau: 0.15, phase: "cruise", minDwell: 0.15}
}

func (f *PhaseFilter) Step(wallSec, aLongG float64) string {
	dt := wallSec - f.prevT
	if dt <= 0 || dt > 1 {
		dt = 0.1
	}
	f.prevT = wallSec
	// однополюсный НЧ y += (dt / (τ+dt)) × (a − y)
	a := dt / (f.tau + dt)
	f.y += a * (aLongG - f.y)

	// Гистерезис на выходе из текущей фазы
	var next string
	switch f.phase {
	case "brake":
		if f.y > -0.18 {
			next = f.classify()
		} else {
			next = "brake"
		}
	case "accel":
		if f.y < 0.12 {
			next = f.classify()
		} else {
			next = "accel"
		}
	default:
		next = f.classify()
	}

	// Min dwell: не меняем фазу чаще чем раз в 150мс
	if next != f.phase && (wallSec-f.since) < f.minDwell {
		return f.phase
	}
	if next != f.phase {
		f.phase = next
		f.since = wallSec
	}
	return f.phase
}

func (f *PhaseFilter) classify() string {
	switch {
	case f.y < -0.25:
		return "brake"
	case f.y < -0.08:
		return "lift"
	case f.y > 0.20:
		return "accel"
	default:
		return "cruise"
	}
}

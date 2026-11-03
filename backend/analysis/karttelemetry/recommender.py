"""Rule-based генератор рекомендаций по v2 spec §10.

Вход — набор поворотных метрик (best + median), секторный разброс,
delta-map (best vs session-N). Выход — упорядоченный список
`Recommendation(text, priority_s, numbers)`, где priority_s — оценка
потерь времени на кругу, numbers — исходные значения в тексте.

Все правила вызываются в контексте одной сессии. Если нет reference
(другой карт/день), часть правил использует только session-median.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class Recommendation:
    text: str
    priority_s: float          # оценка потерь, s
    rule: str                  # id правила ("slip_entry", "lift_extra", ...)
    corner: str | None = None  # к какому Tn относится (если применимо)
    numbers: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Серилизуем численные значения в round'ed form
        d["priority_s"] = round(d["priority_s"], 3)
        d["numbers"] = {k: round(float(v), 3) if np.isfinite(v) else None
                        for k, v in d["numbers"].items()}
        return d


def _median_corner_metric(metrics_by_lap: dict[int, dict[str, dict[str, float]]],
                           corner: str, field_name: str) -> float:
    vals = []
    for lap_m in metrics_by_lap.values():
        if corner in lap_m and np.isfinite(lap_m[corner].get(field_name, np.nan)):
            vals.append(lap_m[corner][field_name])
    return float(np.median(vals)) if vals else float("nan")


def _best_metric_for(best_lap_num: int,
                     metrics_by_lap: dict[int, dict[str, dict[str, float]]],
                     corner: str, field_name: str) -> float:
    if best_lap_num not in metrics_by_lap:
        return float("nan")
    v = metrics_by_lap[best_lap_num].get(corner, {}).get(field_name, float("nan"))
    return float(v)


def generate_recommendations(
    metrics_by_lap: dict[int, dict[str, dict[str, float]]],
    best_lap_num: int,
    delta_map: dict[str, float] | None,
    sector_times_by_lap: dict[int, dict[str, float]] | None,
    reference_metrics: dict[str, dict[str, float]] | None = None,
    hf_session_median: float | None = None,
    valid_laps: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Строит рекомендации по правилам §10.

    `metrics_by_lap[lap_number][corner_name][metric]` — per-lap метрики.
    `delta_map[corner]` — потеря времени на этом повороте (положительная =
      теряем на этом круге относительно session-best по delta map).
    `sector_times_by_lap[lap][sector_name]` — сектор-времена каждого круга.
    `reference_metrics[corner][metric]` — если есть другой карт/день для
      сравнения.
    `hf_session_median` — опционально, медиана hf по всей сессии (для
      "пилёж" правила).
    `valid_laps` — список lap_numbers прошедших gate'ы (NCC + DTW).

    Возвращает list[dict] уже в порядке приоритета (loss_s desc).
    """
    recs: list[Recommendation] = []
    if not metrics_by_lap or best_lap_num not in metrics_by_lap:
        return []

    best_metrics = metrics_by_lap[best_lap_num]
    all_corners = list(best_metrics.keys())
    delta = delta_map or {}

    # медианные значения по поворотам (на валидных кругах)
    med_alat_pk = {c: _median_corner_metric(metrics_by_lap, c, "alat_pk")
                   for c in all_corners}
    med_hf = {c: _median_corner_metric(metrics_by_lap, c, "hf")
              for c in all_corners}
    max_alat_pk = float(np.nanmax([v for v in med_alat_pk.values()
                                    if np.isfinite(v)])) if med_alat_pk else float("nan")

    # ---- 1. Вход на срыве (slip > 0.3s AND brake_len=0 AND Δ > 0.05) ----
    for c in all_corners:
        m = best_metrics[c]
        slip = m.get("slip", float("nan"))
        br = m.get("brake_len", float("nan"))
        d = delta.get(c, 0.0)
        if np.isfinite(slip) and slip > 0.3 and br == 0.0 and d > 0.05:
            recs.append(Recommendation(
                text=f"{c}: вход на скольжении без тормоза — короткое касание "
                     f"тормоза вместо срыва",
                priority_s=max(d, slip * 0.3),
                rule="slip_entry",
                corner=c,
                numbers={"slip_s": slip, "brake_len_s": br, "delta_s": d},
            ))

    # ---- 2. Лишний сброс (lift_len > 0.15s with alat_pk < 0.9·max AND reference не делает) ----
    for c in all_corners:
        m = best_metrics[c]
        ll = m.get("lift_len", 0.0)
        apk = m.get("alat_pk", float("nan"))
        if (ll > 0.15 and np.isfinite(apk) and np.isfinite(max_alat_pk)
                and apk < 0.9 * max_alat_pk):
            # Reference не сбрасывает?
            ref_has_no_lift = False
            if reference_metrics and c in reference_metrics:
                ref_ll = reference_metrics[c].get("lift_len", float("nan"))
                ref_has_no_lift = np.isfinite(ref_ll) and ref_ll < 0.05
            # либо на best-кругу нет lift
            if ref_has_no_lift or ll > 0.3:
                recs.append(Recommendation(
                    text=f"{c}: сброс газа, которого нет на лучших кругах — "
                         f"проезжать на газу",
                    priority_s=max(ll * 0.4, delta.get(c, 0.0)),
                    rule="lift_extra",
                    corner=c,
                    numbers={"lift_len_s": ll, "alat_pk": apk,
                             "max_alat_pk": max_alat_pk},
                ))

    # ---- 3. Потеря на выходе (vmin ≈ ref ±1 km/h BUT Δ > 0.08 AND a_long@after apex < ref) ----
    if reference_metrics:
        for c in all_corners:
            m = best_metrics[c]
            rm = reference_metrics.get(c, {})
            vmin = m.get("vmin_kmh", float("nan"))
            vmin_ref = rm.get("vmin_kmh", float("nan"))
            d = delta.get(c, 0.0)
            if (np.isfinite(vmin) and np.isfinite(vmin_ref)
                    and abs(vmin - vmin_ref) < 1.0 and d > 0.08):
                recs.append(Recommendation(
                    text=f"{c}: минимальная скорость та же, теряешь на выходе — "
                         f"раньше открываться",
                    priority_s=d,
                    rule="exit_loss",
                    corner=c,
                    numbers={"vmin_kmh": vmin, "vmin_ref_kmh": vmin_ref,
                             "delta_s": d},
                ))

    # ---- 4. Пилёж (hf > 1.5× session median AND slip > 0.3) ----
    if hf_session_median is None and med_hf:
        hf_session_median = float(np.nanmedian([v for v in med_hf.values()
                                                 if np.isfinite(v)]))
    if hf_session_median is None or not np.isfinite(hf_session_median):
        hf_session_median = 0.0
    for c in all_corners:
        m = best_metrics[c]
        hf = m.get("hf", float("nan"))
        slip = m.get("slip", 0.0)
        if (np.isfinite(hf) and hf > 1.5 * hf_session_median > 0
                and slip > 0.3):
            recs.append(Recommendation(
                text=f"{c}: много подруливаний и срывов на выходе — спокойнее "
                     f"с газом",
                priority_s=max(slip * 0.3, delta.get(c, 0.0)),
                rule="over_steering",
                corner=c,
                numbers={"hf": hf, "hf_median": hf_session_median,
                         "slip_s": slip},
            ))

    # ---- 5. Повторяемость сектора: σ > 1.5·σ_best OR > 0.10s ----
    if sector_times_by_lap and valid_laps:
        # Собираем сектор-времена через все валидные круги
        sec_vals: dict[str, list[float]] = {}
        for lap in valid_laps:
            if lap not in sector_times_by_lap:
                continue
            for s_name, s_dur in sector_times_by_lap[lap].items():
                if np.isfinite(s_dur):
                    sec_vals.setdefault(s_name, []).append(float(s_dur))
        if sec_vals:
            sigmas = {s: float(np.std(v)) for s, v in sec_vals.items()
                      if len(v) >= 3}
            if sigmas:
                best_sigma = min(sigmas.values())
                for s_name, sg in sigmas.items():
                    if sg > max(1.5 * best_sigma, 0.10):
                        # Проекция σ на loss: σ·k, k≈1 грубо
                        recs.append(Recommendation(
                            text=f"{s_name}: разброс σ={sg*1000:.0f}мс — "
                                 f"тут спрятаны лучшие круги, работать на повторяемость",
                            priority_s=sg,
                            rule="sector_variance",
                            corner=None,
                            numbers={"sigma_s": sg, "best_sigma_s": best_sigma},
                        ))

    # ---- Сортировка по priority (убывание) ----
    recs.sort(key=lambda r: -r.priority_s)
    out = [r.to_dict() for r in recs[:4]]

    return out


def compute_potential_s(sector_times_by_lap: dict[int, dict[str, float]],
                          valid_laps: list[int]) -> float | None:
    """Потенциал на карте: сумма 10-го перцентиля по каждому сектору."""
    if not sector_times_by_lap or not valid_laps:
        return None
    sec_vals: dict[str, list[float]] = {}
    for lap in valid_laps:
        if lap not in sector_times_by_lap:
            continue
        for s_name, s_dur in sector_times_by_lap[lap].items():
            if np.isfinite(s_dur) and 0.1 < s_dur < 30:
                sec_vals.setdefault(s_name, []).append(float(s_dur))
    if not sec_vals:
        return None
    total = 0.0
    for v in sec_vals.values():
        if len(v) < 2:
            continue
        total += float(np.percentile(v, 10))
    return total if total > 0 else None


def compute_trend_s(lap_times_s: list[float], window: int = 10) -> float | None:
    """Усталость/тренд: mean(first N) − mean(last N) валидных кругов. Negative
    = разогнался к концу сессии."""
    if not lap_times_s or len(lap_times_s) < 2 * window:
        return None
    first = float(np.mean(lap_times_s[:window]))
    last = float(np.mean(lap_times_s[-window:]))
    return last - first


__all__ = [
    "Recommendation",
    "generate_recommendations",
    "compute_potential_s",
    "compute_trend_s",
]

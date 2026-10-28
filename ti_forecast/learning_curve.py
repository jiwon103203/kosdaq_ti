"""학습 곡선: 같은 평가구간에서 학습 기간(년)만 바꿔 walk-forward 성과를 비교한다.

매년 첫 표본 날짜에 '최근 ``train_years`` 년'(rolling)으로 다시 학습해 그 해를 예측한다(입력 윈도우 n = s). 평가 첫해는
가장 긴 학습 기간이 표본 안에 들어가는 해로 정하므로, 학습 기간마다 평가일 · 레이블 · Buy & Hold 가 모두 같다.
3 클래스는 No Move 기준값 δ 를 학습구간에서 정하므로 학습 기간마다 평가 레이블이 조금 다르다. ``edge`` 는 같은 레이블의
Buy & Hold 와 비교한 값이라 학습 기간끼리 비교할 수 있다.
학습 기간에 따라 정확도가 오르면 데이터가 더 있으면 나아질 여지가 있고, 평평하면(특히 Buy & Hold 대비 차이가 잡음 범위
안이면) 데이터를 늘리는 것만으로는 나아지지 않는다는 뜻이다.

표본 수 기준 (근사): 정확도 p 가 기준 p0 보다 d 높은지를 유의수준 α(양측) · 검정력 1-β 로 확인하려면 독립 표본이
(z_{α/2} + z_β)² · p(1-p) / d² 개 필요하다. 겹치는 s기간 레이블은 독립 표본이 약 N/s 개다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .experiment import (PERIODS_PER_YEAR, Job, Prepared, paper_metrics, refit_dates, walk_forward_jobs,
                         walk_forward_predictions)

Z_ALPHA = {0.05: 1.959964, 0.10: 1.644854, 0.01: 2.575829}  # 양측 유의수준 → z
Z_POWER = {0.8: 0.841621, 0.9: 1.281552, 0.5: 0.0}          # 검정력 → z


def required_samples(edge: float, p: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> float:
    """정확도를 기준보다 ``edge`` 만큼 높게 맞히는지 확인하는 데 필요한 독립 표본 수 (정규근사)."""
    z = Z_ALPHA[alpha] + Z_POWER[power]
    return z * z * p * (1.0 - p) / (edge * edge)


def min_detectable_edge(n_independent: float, p: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> float:
    """독립 표본 ``n_independent`` 개로 확인할 수 있는 최소 정확도 차이 (``required_samples`` 의 역)."""
    z = Z_ALPHA[alpha] + Z_POWER[power]
    return z * np.sqrt(p * (1.0 - p) / max(n_independent, 1.0))


def learning_curve_jobs(p: Prepared, models: Sequence[str], horizons: Sequence[int], class_counts: Sequence[int],
                        first_year: int, windows: Sequence[int], train_years: Sequence[int]) -> List[Job]:
    """학습 기간마다 walk-forward 작업. key 와 meta 에 학습 기간을 붙여 한 캐시 폴더에서 구분한다."""
    jobs = []
    for ty in train_years:
        for job in walk_forward_jobs(p, models, horizons, class_counts, first_year, windows, ty):
            job.key = f"{job.key}_ty{ty}"
            job.meta = {**job.meta, "train_years": ty}
            jobs.append(job)
    return jobs


def learning_curve_predictions(jobs: Sequence[Job], results: Dict, prepared: Dict[str, Prepared]) -> pd.DataFrame:
    frames = []
    for ty in sorted({j.meta["train_years"] for j in jobs}):
        sub = [j for j in jobs if j.meta["train_years"] == ty]
        frames.append(walk_forward_predictions(sub, results, prepared).assign(train_years=ty))
    return pd.concat(frames, ignore_index=True)


def learning_curve_metrics(pred: pd.DataFrame, jobs: Sequence[Job], results: Dict,
                           periods: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    """(학습 기간, 모델, 클래스 수, 예측기간)마다 정확도 · Buy & Hold 대비 차이 · 표준오차 · 학습 표본 · CV 정확도.

    ``edge`` = 정확도 - Buy & Hold(항상 Up) 정확도, ``mde`` = 이 평가구간으로 확인할 수 있는 최소 차이(α 5%, 검정력
    80%), ``cv_gap`` = 학습구간 교차검증 정확도 - 평가 정확도 (클수록 하이퍼파라미터 선택 · 과적합으로 부풀었다는 뜻).
    """
    info = pd.DataFrame([{"train_years": j.meta["train_years"], "model": j.model, "n_classes": j.meta["n_classes"],
                          "horizon": j.meta["horizon"], "n_train": results[j.key][1].get("n_train", np.nan),
                          "cv_acc": results[j.key][1].get("cv_acc", np.nan)} for j in jobs])
    fit = info.groupby(["train_years", "model", "n_classes", "horizon"])[["n_train", "cv_acc"]].mean()
    pred = pred[pred["label"].notna()]
    rows = []
    for (ty, m, k, s), g in pred.groupby(["train_years", "model", "n_classes", "horizon"]):
        y, r, s = g["label"].to_numpy().astype(int), g["fwd_ret"].to_numpy(), int(s)
        pm = paper_metrics(g["pred"].to_numpy(), y, r, s, periods)
        bh = paper_metrics(np.ones(len(g), dtype=int), y, r, s, periods)
        n_ind = len(g) / s
        rows.append({"train_years": ty, "model": m, "n_classes": k, "horizon": s,
                     "start": g["date"].min().date(), "end": g["date"].max().date(), "n_eval": len(g),
                     "n_eval_independent": n_ind, "accuracy": pm["accuracy"], "bh_accuracy": bh["accuracy"],
                     "edge": pm["accuracy"] - bh["accuracy"], "acc_se": pm["acc_se"],
                     "mde": min_detectable_edge(n_ind, bh["accuracy"]), "sharpe": pm["sharpe"],
                     "bh_sharpe": bh["sharpe"], "frac_up": pm["frac_up"],
                     **fit.loc[(ty, m, k, s)].to_dict()})
    out = pd.DataFrame(rows)
    out["n_train_independent"] = out["n_train"] / out["horizon"]
    out["cv_gap"] = out["cv_acc"] - out["accuracy"]
    return out


def learning_curve_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """예측기간 평균: 학습 기간 × 모델 × 클래스 수."""
    cols = ["accuracy", "bh_accuracy", "edge", "mde", "cv_acc", "cv_gap", "n_train", "n_train_independent"]
    g = metrics.groupby(["n_classes", "model", "train_years"])
    out = g[cols].mean()
    out["beats_bh"] = g["edge"].apply(lambda e: float(np.mean(e > 0)))         # 차이가 양수인 예측기간 비율
    out["significant"] = (metrics.assign(sig=metrics["edge"] > metrics["mde"])  # 확인 가능한 만큼 높은 예측기간 비율
                          .groupby(["n_classes", "model", "train_years"])["sig"].mean())
    return out.reset_index()


def default_first_year(dates: pd.DatetimeIndex, max_train_years: int) -> Optional[int]:
    """가장 긴 학습 기간이 표본 안에 다 들어가는 첫 재학습 연도 (재학습일 - 학습 기간 ≥ 첫 표본 날짜). 없으면 None."""
    for start in refit_dates(dates, dates[0].year):
        if start - pd.DateOffset(years=max_train_years) >= dates[0]:
            return start.year
    return None


def sample_size_table(edges: Sequence[float], horizons: Sequence[int], periods: int,
                      p: float = 0.5) -> pd.DataFrame:
    """엣지 d 를 확인하는 데 필요한 독립 표본 수와, 예측기간 s 마다 그만큼 쌓는 데 걸리는 평가 기간(년)."""
    rows = []
    for d in edges:
        n = required_samples(d, p)
        rows.append({"edge_pp": d * 100, "independent_samples": n,
                     **{f"years_s{s}": n * s / periods for s in horizons}})
    return pd.DataFrame(rows)

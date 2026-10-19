"""레이블 · 학습/평가 분할 · 논문 성과지표 · 포트폴리오 백테스트 · 그리드 실험 · walk-forward.

시나리오 (같은 평가구간에서 비교)
    ohlc_short   OHLC 지표, 시가·고가·저가가 있는 날부터 학습           ← 논문 그대로
    close_short  종가 전용 지표, ohlc_short 와 같은 날짜                 ← 지표 종류만 바꾼 효과
    close_long   종가 전용 지표, 종가가 있는 첫날부터 학습              ← 학습 기간을 늘린 효과

그리드 (논문 Table III–X): 평가구간 = OHLC 표본의 마지막 ``test_frac`` (논문 2:1). 예측기간 s × 입력 윈도우 n ×
2/3 클래스 × 모델마다 한 번 학습해 평가구간 전체를 예측한다. 학습 표본은 레이블 끝 날짜(t+s)가 평가 시작 전인 것만 쓴다.

walk-forward: 매년 첫 거래일에 그 전까지 레이블이 확정된 표본으로 다시 학습해 그 해를 예측한다. 입력 윈도우는 논문의
결론대로 n = s (s < 3 이면 3).

논문 성과지표 (Sec. 6): 매 평가일에 'Up' → s일 매수, 'Down' → s일 매도, 'No Move' → 거래 없음.
    정확도, 승률(수익 거래 비율), 거래당 수익률, Sharpe = sqrt(252/s) · 평균 / 표준편차 (무위험금리 0).
겹치는 s일 거래를 독립으로 세므로 낙관적이다. 그래서 매일 신호의 1/s 를 s일 보유하는 tranche 포트폴리오로
일간 수익률을 만들어(진입 지연 · 거래비용 포함) CAGR · Sharpe · MDD 도 계산한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import ohlc_start
from .indicators import compute_indicators
from .models import fit_predict

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 5, 7, 10, 15, 20, 25, 30)
WINDOWS = (3, 5, 7, 10, 15, 20, 25, 30)
PERIODS_PER_YEAR = 252
SCENARIOS = ("ohlc_short", "close_short", "close_long")


# ---------------------------------------------------------------------------
# 레이블
# ---------------------------------------------------------------------------

def forward_returns(close: pd.Series, horizons: Iterable[int]) -> pd.DataFrame:
    """열 s = (C_{t+s} - C_t) / C_t. 끝쪽 s 일은 NaN."""
    return pd.DataFrame({s: close.shift(-s) / close - 1.0 for s in horizons}, index=close.index)


def label_end_dates(index: pd.DatetimeIndex, s: int) -> pd.Series:
    """t 일 s일 레이블이 확정되는 날짜 (t+s 거래일). 끝쪽은 NaT."""
    return pd.Series(index, index=index).shift(-s)


def no_move_threshold(r_train: np.ndarray) -> float:
    """3 클래스 기준값 δ: 학습구간 |r| 의 1/3 분위수 → 'No Move' 가 약 1/3 (논문 Table I 과 같은 기준)."""
    return float(np.quantile(np.abs(r_train), 1.0 / 3.0))


def make_labels(r: np.ndarray, n_classes: int, delta: float = 0.0) -> np.ndarray:
    """2 클래스: Up(1) r > 0, Down(-1) r <= 0 (논문 식 1). 3 클래스: ±δ 밖이면 Up/Down, 안이면 No Move(0) (식 2)."""
    if n_classes == 2:
        return np.where(r > 0, 1, -1)
    if n_classes == 3:
        return np.where(r > delta, 1, np.where(r < -delta, -1, 0))
    raise ValueError(f"n_classes must be 2 or 3, got {n_classes}")


# ---------------------------------------------------------------------------
# 성과
# ---------------------------------------------------------------------------

def paper_metrics(pred: np.ndarray, label: np.ndarray, fwd_ret: np.ndarray, horizon: int) -> Dict[str, float]:
    """논문 식 15–20: 정확도 · 거래 수 · 승률 · 거래당 수익률 · Sharpe (s일 거래를 연 252/s 회로 연율화).

    ``acc_se`` 는 겹치는 s일 레이블을 감안한 정확도 표준오차 근사 sqrt(p(1-p) / (N/s)) — 독립 표본이 N/s 개쯤이다.
    """
    trade = pred != 0
    pnl = np.sign(pred[trade]) * fwd_ret[trade]
    sd = pnl.std(ddof=1) if len(pnl) > 1 else np.nan
    acc = float(np.mean(pred == label))
    return {
        "accuracy": acc,
        "acc_se": float(np.sqrt(acc * (1.0 - acc) / max(len(pred) / horizon, 1.0))),
        "n_trades": int(trade.sum()),
        "win_rate": float(np.mean(pnl > 0)) if len(pnl) else np.nan,
        "ret_per_trade": float(pnl.mean()) if len(pnl) else np.nan,
        "sharpe": float(np.sqrt(PERIODS_PER_YEAR / horizon) * pnl.mean() / sd) if sd and sd > 0 else np.nan,
        "frac_up": float(np.mean(pred == 1)),
    }


def max_drawdown(ret: pd.Series) -> float:
    wealth = np.concatenate([[1.0], (1.0 + ret.to_numpy()).cumprod()])
    return float((wealth / np.maximum.accumulate(wealth) - 1.0).min())


def performance_metrics(ret: pd.Series, weight: Optional[pd.Series] = None,
                        periods: int = PERIODS_PER_YEAR) -> Dict[str, float]:
    """일간 수익률의 CAGR · 변동성 · Sharpe · Sortino · MDD · Calmar (무위험금리 0) (+ 비중 평균 · 회전율)."""
    ret = ret.dropna()
    n = len(ret)
    if n == 0:
        return {}
    years = n / periods
    total = float((1.0 + ret).prod() - 1.0)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    std = ret.std()
    downside = np.sqrt(np.mean(np.minimum(ret.to_numpy(), 0.0) ** 2))
    mdd = max_drawdown(ret)
    m = {
        "start": ret.index[0].date().isoformat(),
        "end": ret.index[-1].date().isoformat(),
        "n_days": n,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": float(std * np.sqrt(periods)),
        "sharpe": float(ret.mean() / std * np.sqrt(periods)) if std > 0 else np.nan,
        "sortino": float(ret.mean() / downside * np.sqrt(periods)) if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }
    if weight is not None:
        w = weight.reindex(ret.index)
        dw = w.diff().abs().fillna(0.0)
        m["avg_weight"] = float(w.mean())
        m["n_trades"] = int((dw > 1e-12).sum())
        m["turnover"] = float(dw.sum() / years)
    return m


def tranche_backtest(signal: pd.Series, close: pd.Series, horizon: int, delay: int = 1, cost_bps: float = 10.0,
                     long_only: bool = False) -> pd.DataFrame:
    """매일 신호(+1/0/-1)의 1/s 를 s일 보유하는 tranche 포트폴리오의 일간 수익률.

    ``signal`` 은 t 일 종가로 만든 신호, ``delay`` 일 뒤 종가에 진입한다(0 이면 논문처럼 t 일 종가).
    t 일 종가 ~ t+1 일 종가 보유 비중 = 최근 s 개 (지연된) 신호의 평균. 비중 변화 1 당 ``cost_bps`` 를 뺀다.
    반환 열: ``weight`` (그날 수익률에 적용된 비중), ``asset_ret``, ``strat_ret``. 첫날(전일 비중 없음)은 뺀다.
    """
    sig = signal.clip(lower=0) if long_only else signal
    pos = (sig.rolling(horizon, min_periods=1).sum() / horizon).shift(delay).fillna(0.0)
    asset_ret = close.reindex(signal.index).pct_change()
    held = pos.shift(1)
    cost = (pos - pos.shift(1).fillna(0.0)).abs().shift(1) * cost_bps / 1e4
    out = pd.DataFrame({"weight": held, "asset_ret": asset_ret, "strat_ret": held * asset_ret - cost})
    return out.iloc[1:]


# ---------------------------------------------------------------------------
# 시나리오 · 데이터 준비
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    close_only: bool
    start: pd.Timestamp  # 이 날짜부터의 가격만 쓴다


def build_scenarios(df: pd.DataFrame, names: Sequence[str] = SCENARIOS) -> List[Scenario]:
    o = ohlc_start(df)
    known = {"close_long": Scenario("close_long", True, df.index[0])}
    if o is not None:
        known["ohlc_short"] = Scenario("ohlc_short", False, o)
        known["close_short"] = Scenario("close_short", True, o)
    missing = [n for n in names if n not in known]
    if missing:
        raise ValueError(f"시나리오 {missing} 를 만들 수 없습니다 (OHLC 없음). 가능한 시나리오: {list(known)}")
    return [known[n] for n in names]


@dataclass
class Prepared:
    scenario: Scenario
    close: pd.Series                 # 시나리오 구간 종가
    features: Dict[int, pd.DataFrame]  # 입력 윈도우 n → 지표 (표본 날짜만)
    dates: pd.DatetimeIndex          # 모든 윈도우 지표가 있는 표본 날짜 (warmup 이후)
    fwd: pd.DataFrame                # 표본 날짜의 s일 선행 수익률


def prepare(df: pd.DataFrame, sc: Scenario, windows: Sequence[int], horizons: Sequence[int],
            warmup: int = 120) -> Prepared:
    """시나리오 구간 가격으로 지표를 만들고, 앞 ``warmup`` 일(EWM 초기값 영향)을 버린다."""
    px = df.loc[sc.start:]
    feats = {n: compute_indicators(px, n, sc.close_only) for n in windows}
    valid = pd.concat([f.notna().all(axis=1) for f in feats.values()], axis=1).all(axis=1)
    dates = px.index[warmup:]
    dates = dates[valid.reindex(dates).to_numpy()]
    return Prepared(sc, px["close"], {n: f.loc[dates] for n, f in feats.items()}, dates,
                    forward_returns(px["close"], horizons).loc[dates])


def _standardize(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """학습구간 평균·표준편차로 표준화 (논문 Sec. 4.2)."""
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def training_rows(p: Prepared, s: int, cutoff: pd.Timestamp, first: Optional[pd.Timestamp] = None) -> pd.DatetimeIndex:
    """레이블이 ``cutoff`` 전에 확정되는 표본 날짜 (레이블 구간이 평가구간에 걸치지 않게)."""
    end = label_end_dates(p.close.index, s).reindex(p.dates)
    keep = (end < cutoff).to_numpy()
    if first is not None:
        keep &= (p.dates >= first)
    return p.dates[keep]


def matched_window(s: int, windows: Sequence[int] = WINDOWS) -> int:
    """논문 결론의 n ≈ s: 후보 중 s 에 가장 가까운 윈도우 (동점이면 작은 쪽)."""
    return min(windows, key=lambda n: (abs(n - s), n))


# ---------------------------------------------------------------------------
# 병렬 실행 · 캐시
# ---------------------------------------------------------------------------

@dataclass
class Job:
    key: str
    model: str
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_te: np.ndarray
    gap: int
    meta: Dict


def _run_job(job: Job, model_kwargs: Dict, cache_dir: Optional[str]):
    path = os.path.join(cache_dir, job.key + ".pkl") if cache_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return job.key, pickle.load(f)
    t0 = time.time()
    pred, info = fit_predict(job.model, job.X_tr, job.y_tr, job.X_te, job.gap, **model_kwargs.get(job.model, {}))
    result = (pred, {**{k: float(v) for k, v in info.items()}, "seconds": time.time() - t0, "n_train": len(job.y_tr)})
    if path:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(result, f)
        os.replace(tmp, path)
    return job.key, result


def run_jobs(jobs: List[Job], n_jobs: int = 1, model_kwargs: Optional[Dict] = None,
             cache_dir: Optional[str] = None, label: str = "jobs") -> Dict[str, Tuple[np.ndarray, Dict]]:
    """SVM 처럼 오래 걸리는 작업부터 병렬로 돌린다. ``cache_dir`` 가 있으면 끝난 작업을 저장·재사용한다."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    order = {"svm": 0, "ann": 1, "knn": 2}
    jobs = sorted(jobs, key=lambda j: (order.get(j.model, 3), -len(j.y_tr)))
    out, t0 = {}, time.time()
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_run_job)(job, model_kwargs or {}, cache_dir) for job in jobs)
    for i, (key, res) in enumerate(results, 1):
        out[key] = res
        if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
            logger.info("%s: %d/%d done (%.0fs)", label, i, len(jobs), time.time() - t0)
    return out


def config_hash(cfg: Dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# 그리드 실험 (논문 Table III–X)
# ---------------------------------------------------------------------------

def common_test_dates(prepared: Sequence[Prepared], horizons: Sequence[int], test_frac: float) -> pd.DatetimeIndex:
    """가장 늦게 시작하는 시나리오의 (최장 예측기간 레이블이 있는) 표본 중 마지막 ``test_frac``."""
    ref = max(prepared, key=lambda p: p.dates[0])
    labeled = ref.dates[ref.fwd[max(horizons)].notna().to_numpy()]
    test = labeled[int(round(len(labeled) * (1.0 - test_frac))):]
    for p in prepared:
        if not test.isin(p.dates).all():
            raise ValueError(f"{p.scenario.name}: 평가구간 날짜 일부에 지표가 없습니다")
    return test


def grid_jobs(prepared: Sequence[Prepared], test: pd.DatetimeIndex, models: Sequence[str],
              horizons: Sequence[int], windows: Sequence[int], class_counts: Sequence[int]) -> Tuple[List[Job], Dict]:
    jobs, thresholds = [], {}
    for p in prepared:
        name = p.scenario.name
        for s in horizons:
            tr_dates = training_rows(p, s, test[0])
            r_tr, r_te = p.fwd.loc[tr_dates, s].to_numpy(), p.fwd.loc[test, s].to_numpy()
            delta = no_move_threshold(r_tr)
            thresholds[(name, s)] = {"delta": delta, "train_start": tr_dates[0], "train_end": tr_dates[-1],
                                     "n_train": len(tr_dates)}
            for n in windows:
                X_tr, X_te = _standardize(p.features[n].loc[tr_dates].to_numpy(), p.features[n].loc[test].to_numpy())
                for k in class_counts:
                    y_tr = make_labels(r_tr, k, delta)
                    for m in models:
                        jobs.append(Job(f"grid_{name}_{m}_c{k}_s{s}_n{n}", m, X_tr, y_tr, X_te, s - 1,
                                        {"scenario": name, "model": m, "n_classes": k, "horizon": s, "window": n,
                                         "delta": delta, "y_te": make_labels(r_te, k, delta), "r_te": r_te}))
    return jobs, thresholds


def grid_metrics(jobs: Sequence[Job], results: Dict, test: pd.DatetimeIndex) -> pd.DataFrame:
    """설정마다 논문 지표 + 같은 평가구간의 Buy & Hold (항상 'Up')."""
    rows, seen = [], set()
    for job in jobs:
        meta = job.meta
        pred, info = results[job.key]
        row = {k: meta[k] for k in ("scenario", "model", "n_classes", "horizon", "window")}
        rows.append({**row, **paper_metrics(pred, meta["y_te"], meta["r_te"], meta["horizon"]), **info})
        bh_key = (meta["scenario"], meta["n_classes"], meta["horizon"])
        if bh_key not in seen:
            seen.add(bh_key)
            bh = paper_metrics(np.ones(len(test), dtype=int), meta["y_te"], meta["r_te"], meta["horizon"])
            rows.append({**row, "model": "buy_hold", "window": np.nan, **bh})
    return pd.DataFrame(rows)


def grid_predictions(jobs: Sequence[Job], results: Dict, test: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for job in jobs:
        m = job.meta
        frames.append(pd.DataFrame({"date": test, "scenario": m["scenario"], "model": m["model"],
                                    "n_classes": m["n_classes"], "horizon": m["horizon"], "window": m["window"],
                                    "pred": results[job.key][0], "label": m["y_te"], "fwd_ret": m["r_te"]}))
    return pd.concat(frames, ignore_index=True)


def best_by_horizon(metrics: pd.DataFrame, columns=("accuracy", "win_rate", "ret_per_trade", "sharpe")) -> pd.DataFrame:
    """논문 Table X: 예측기간마다 입력 윈도우 중 최고값 (Buy & Hold 는 그대로)."""
    keys = ["scenario", "n_classes", "horizon", "model"]
    return metrics.groupby(keys)[list(columns)].max().unstack("model")


def window_pattern(metrics: pd.DataFrame, columns=("accuracy", "win_rate", "ret_per_trade", "sharpe"),
                   windows: Sequence[int] = WINDOWS) -> pd.DataFrame:
    """'n ≈ s 일 때 가장 좋다' 패턴 점검.

    best_window      예측기간마다 지표가 가장 높은 입력 윈도우 (예측기간 순서대로)
    spearman         예측기간과 best_window 의 순위상관 (논문 패턴이면 양수, 1 에 가까움)
    matched_rank     n ≈ s 칸의 윈도우 내 순위 평균 (1 = 최고, len(windows) = 최저, 무작위면 중간)
    matched_minus_avg  n ≈ s 칸 값 - 같은 예측기간 윈도우 평균, 예측기간 평균
    """
    rows = []
    m = metrics[metrics["model"] != "buy_hold"]
    for key, g in m.groupby(["scenario", "model", "n_classes"]):
        for col in columns:
            table = g.pivot(index="horizon", columns="window", values=col)
            best = table.idxmax(axis=1)
            matched = [matched_window(s, list(table.columns)) for s in table.index]
            ranks = table.rank(axis=1, ascending=False)
            rows.append({
                "scenario": key[0], "model": key[1], "n_classes": key[2], "metric": col,
                "best_window": ",".join(str(int(b)) for b in best),
                "spearman": float(pd.Series(best.to_numpy(), dtype=float).corr(
                    pd.Series(table.index.to_numpy(), dtype=float), method="spearman"))
                if best.nunique() > 1 else np.nan,  # 최고 윈도우가 전부 같으면 정의되지 않음
                "matched_rank": float(np.mean([ranks.loc[s, n] for s, n in zip(table.index, matched)])),
                "matched_minus_avg": float(np.mean([table.loc[s, n] - table.loc[s].mean()
                                                    for s, n in zip(table.index, matched)])),
            })
    return pd.DataFrame(rows)


def portfolio_table(pred: pd.DataFrame, close: pd.Series, delay: int, cost_bps: float,
                    keys=("scenario", "model", "n_classes", "horizon", "window")) -> pd.DataFrame:
    """설정마다 tranche 포트폴리오(롱숏 · 롱온리) 성과 + 시나리오별 같은 기간 Buy & Hold."""
    rows = []
    for key, g in pred.groupby(list(keys)):
        sig = pd.Series(g["pred"].to_numpy(), index=pd.DatetimeIndex(g["date"])).sort_index()
        s = int(key[list(keys).index("horizon")])
        for mode, long_only in (("long_short", False), ("long_only", True)):
            bt = tranche_backtest(sig, close, s, delay=delay, cost_bps=cost_bps, long_only=long_only)
            rows.append({**dict(zip(keys, key)), "mode": mode,
                         **performance_metrics(bt["strat_ret"], weight=bt["weight"])})
    for name, g in pred.groupby("scenario"):
        dates = pd.DatetimeIndex(np.sort(g["date"].unique()))
        bh = close.reindex(dates).pct_change().iloc[1:]
        rows.append({"scenario": name, "model": "buy_hold", "mode": "long_only",
                     **performance_metrics(bh, weight=pd.Series(1.0, index=bh.index))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# walk-forward (n = s, 매년 재학습)
# ---------------------------------------------------------------------------

def refit_dates(dates: pd.DatetimeIndex, first_year: int) -> List[pd.Timestamp]:
    """``first_year`` 부터 매년 첫 표본 날짜."""
    years = sorted({d.year for d in dates if d.year >= first_year})
    return [dates[dates.year == y][0] for y in years]


def walk_forward_jobs(p: Prepared, models: Sequence[str], horizons: Sequence[int], class_counts: Sequence[int],
                      first_year: int, windows: Sequence[int] = WINDOWS,
                      train_years: Optional[int] = None) -> List[Job]:
    """매년 재학습 작업. ``train_years`` 가 있으면 최근 그 기간만(rolling), 없으면 전체(expanding)."""
    jobs = []
    refits = refit_dates(p.dates, first_year)
    for i, start in enumerate(refits):
        end = refits[i + 1] if i + 1 < len(refits) else p.dates[-1] + pd.Timedelta(days=1)
        te_dates = p.dates[(p.dates >= start) & (p.dates < end)]
        for s in horizons:
            n = matched_window(s, windows)
            first = start - pd.DateOffset(years=train_years) if train_years else None
            tr_dates = training_rows(p, s, start, first)
            r_tr = p.fwd.loc[tr_dates, s].to_numpy()
            delta = no_move_threshold(r_tr)
            X_tr, X_te = _standardize(p.features[n].loc[tr_dates].to_numpy(), p.features[n].loc[te_dates].to_numpy())
            for k in class_counts:
                y_tr = make_labels(r_tr, k, delta)
                for m in models:
                    key = f"wf_{p.scenario.name}_{m}_c{k}_s{s}_n{n}_{start.year}"
                    jobs.append(Job(key, m, X_tr, y_tr, X_te, s - 1,
                                    {"scenario": p.scenario.name, "model": m, "n_classes": k, "horizon": s,
                                     "window": n, "delta": delta, "dates": te_dates, "refit": start}))
    return jobs


def walk_forward_predictions(jobs: Sequence[Job], results: Dict, prepared: Dict[str, Prepared]) -> pd.DataFrame:
    """walk-forward 예측을 이어 붙이고, 그 해 학습 기준값 δ 로 실제 레이블을 단다 (끝쪽 레이블 없는 날은 NaN)."""
    frames = []
    for job in jobs:
        m = job.meta
        r = prepared[m["scenario"]].fwd.loc[m["dates"], m["horizon"]].to_numpy()
        label = make_labels(r, m["n_classes"], m["delta"]).astype(float)
        label[np.isnan(r)] = np.nan
        frames.append(pd.DataFrame({"date": m["dates"], "scenario": m["scenario"], "model": m["model"],
                                    "n_classes": m["n_classes"], "horizon": m["horizon"], "window": m["window"],
                                    "refit": m["refit"], "pred": results[job.key][0], "label": label,
                                    "fwd_ret": r}))
    return pd.concat(frames, ignore_index=True).sort_values(["scenario", "model", "n_classes", "horizon", "date"])


def walk_forward_metrics(pred: pd.DataFrame, start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """walk-forward 예측의 논문 지표 (레이블이 확정된 날만) + Buy & Hold."""
    if start is not None:
        pred = pred[pred["date"] >= start]
    pred = pred[pred["label"].notna()]
    rows = []
    for key, g in pred.groupby(["scenario", "model", "n_classes", "horizon", "window"]):
        y, r, s = g["label"].to_numpy().astype(int), g["fwd_ret"].to_numpy(), int(key[3])
        base = dict(zip(["scenario", "model", "n_classes", "horizon", "window"], key))
        rows.append({**base, "start": g["date"].min().date(), "end": g["date"].max().date(),
                     **paper_metrics(g["pred"].to_numpy(), y, r, s)})
        rows.append({**base, "model": "buy_hold", "start": g["date"].min().date(), "end": g["date"].max().date(),
                     **paper_metrics(np.ones(len(g), dtype=int), y, r, s)})
    return pd.DataFrame(rows).drop_duplicates(subset=["scenario", "model", "n_classes", "horizon"])

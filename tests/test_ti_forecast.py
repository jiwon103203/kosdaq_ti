"""ti_forecast (기술지표 윈도우 × 예측기간 실험) 테스트 — 지표 식, 미래 정보 미사용, 분할·백테스트 규칙.

    pytest tests/test_ti_forecast.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_ti_forecast  # noqa: E402
from ti_forecast.data import load_ohlc, ohlc_start  # noqa: E402
from ti_forecast.experiment import (Job, Scenario, build_scenarios, cache_suffix, common_test_dates,  # noqa: E402
                                    grid_jobs, make_labels, matched_window, no_move_threshold, paper_metrics,
                                    prepare, price_inputs, refit_dates, run_jobs, training_rows, tranche_backtest,
                                    walk_forward_jobs)
from ti_forecast.indicators import NAMES, compute_indicators  # noqa: E402
from ti_forecast.models import (_svm_direct, arma_cum_forecast, check_dependencies, fit_predict,  # noqa: E402
                                fit_predict_svm, purged_kfold, select_arma)


def simulate_ohlc(n=900, seed=0, start="2012-01-02", ohlc_from=300):
    """랜덤워크 종가 + 그 주변 시가·고가·저가. 앞 ``ohlc_from`` 일은 종가만 있다."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    df.iloc[:ohlc_from, :3] = np.nan
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def test_close_only_is_ohlc_formula_with_high_low_equal_close():
    df = simulate_ohlc(ohlc_from=0)
    flat = df.assign(high=df["close"], low=df["close"])
    for n in (3, 10):
        pd.testing.assert_frame_equal(compute_indicators(flat, n), compute_indicators(df, n, close_only=True))


def test_indicators_use_only_past_prices():
    df = simulate_ohlc(ohlc_from=0)
    cut = 500
    shocked = df.copy()
    shocked.iloc[cut + 1:] *= 1.5
    for close_only in (False, True):
        a = compute_indicators(df, 7, close_only).iloc[:cut + 1]
        b = compute_indicators(shocked, 7, close_only).iloc[:cut + 1]
        pd.testing.assert_frame_equal(a, b)


def test_indicator_values_on_rising_series():
    idx = pd.bdate_range("2020-01-01", periods=60)
    df = pd.DataFrame({"close": np.arange(100.0, 160.0)}, index=idx)
    f = compute_indicators(df, 5, close_only=True).iloc[-1]
    assert list(f.index) == list(NAMES)
    assert f["rsi"] == pytest.approx(100.0)
    assert f["stoch_k"] == pytest.approx(100.0)
    assert f["willr"] == pytest.approx(0.0)
    assert f["adx"] == pytest.approx(100.0)
    assert f["roc"] == pytest.approx(159.0 / 154.0 - 1.0)
    assert f["sma"] == pytest.approx(159.0 / 157.0 - 1.0)
    assert f["cci"] > 0


def test_stochastic_k_and_williams_r_are_complements():
    f = compute_indicators(simulate_ohlc(ohlc_from=0), 10).dropna()
    np.testing.assert_allclose(f["stoch_k"] - f["willr"], 100.0)  # %R 은 TA-Lib 처럼 음수 척도


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def test_load_dataguide_layout(tmp_path):
    df = simulate_ohlc(n=50, ohlc_from=10)
    meta = [["Refresh", "Last Update", None, None, None], ["Code", "IKQ150", "IKQ150", "IKQ150", "IKQ150"],
            ["Name", "코스닥 150", "코스닥 150", "코스닥 150", "코스닥 150"], [None] * 5,
            ["D A T E", "저가지수", "고가지수", "시가지수", "종가지수"]]
    body = [[d, r.low, r.high, r.open, r.close] for d, r in df.iterrows()]
    body.append([pd.Timestamp("2099-01-01"), None, None, None, None])  # 장중이라 비어 있는 마지막 행
    path = tmp_path / "dg.xlsx"
    pd.DataFrame(meta + body).to_excel(path, header=False, index=False)
    out = load_ohlc(str(path))
    assert len(out) == 50
    np.testing.assert_allclose(out["close"], df["close"])
    np.testing.assert_allclose(out["high"].dropna(), df["high"].dropna())
    assert ohlc_start(out) == df.index[10]


def test_load_plain_csv(tmp_path):
    df = simulate_ohlc(n=30, ohlc_from=0)
    path = tmp_path / "px.csv"
    df.reset_index().to_csv(path, index=False)
    out = load_ohlc(str(path))
    np.testing.assert_allclose(out[["open", "high", "low", "close"]], df)


def test_scenarios_follow_ohlc_availability():
    df = simulate_ohlc()
    names = {s.name: s for s in build_scenarios(df)}
    assert names["close_long"].start == df.index[0]
    assert names["ohlc_short"].start == names["close_short"].start == df.index[300]
    with pytest.raises(ValueError):
        build_scenarios(df.assign(open=np.nan), ("ohlc_short",))


# ---------------------------------------------------------------------------
# 레이블 · 분할
# ---------------------------------------------------------------------------

def test_labels_and_no_move_threshold():
    r = np.array([-0.03, -0.001, 0.0, 0.002, 0.05])
    np.testing.assert_array_equal(make_labels(r, 2), [-1, -1, -1, 1, 1])
    np.testing.assert_array_equal(make_labels(r, 3, 0.01), [-1, 0, 0, 0, 1])
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 30000)
    assert np.mean(make_labels(x, 3, no_move_threshold(x)) == 0) == pytest.approx(1 / 3, abs=0.01)


def test_purged_kfold_drops_overlapping_labels():
    n, gap = 100, 4
    for train, val in purged_kfold(n, 5, gap):
        assert len(np.intersect1d(train, val)) == 0
        # 레이블 (i, i+gap+1] 과 검증 레이블이 겹치는 학습 표본이 없어야 한다
        assert np.all((train <= val[0] - gap - 1) | (train >= val[-1] + gap + 1))
    covered = np.concatenate([v for _, v in purged_kfold(n, 5, 0)])
    np.testing.assert_array_equal(covered, np.arange(n))


def test_training_rows_labels_end_before_cutoff():
    df = simulate_ohlc()
    p = prepare(df, Scenario("close_long", True, df.index[0]), (3, 5), (1, 10), warmup=40)
    cutoff = p.dates[400]
    for s in (1, 10):
        rows = training_rows(p, s, cutoff)
        last = p.close.index.get_loc(rows[-1])
        assert p.close.index[last + s] < cutoff
        assert p.close.index[last + s + 1] >= cutoff  # 쓸 수 있는 마지막 표본까지는 쓴다


def test_common_test_dates_are_last_fraction_of_shortest_sample():
    df = simulate_ohlc()
    prepared = [prepare(df, sc, (3, 5), (1, 5), warmup=40) for sc in build_scenarios(df)]
    test = common_test_dates(prepared, (1, 5), 1 / 3)
    short = prepared[0]
    labeled = short.dates[short.fwd[5].notna().to_numpy()]
    assert test[-1] == labeled[-1]
    assert len(test) == pytest.approx(len(labeled) / 3, abs=1)


def test_walk_forward_refits_each_year_without_leakage():
    df = simulate_ohlc(n=1200)
    p = prepare(df, Scenario("close_long", True, df.index[0]), (3, 5), (1, 5), warmup=40)
    refits = refit_dates(p.dates, 2014)
    assert all(r.month == 1 for r in refits) and refits[0].year == 2014
    jobs = walk_forward_jobs(p, ("knn",), (1, 5), (2,), 2014, windows=(3, 5))
    assert len(jobs) == len(refits) * 2
    for job in jobs:
        assert job.meta["window"] == matched_window(job.meta["horizon"], (3, 5))
        assert job.meta["dates"][0] >= job.meta["refit"]
    assert matched_window(1) == 3 and matched_window(12) == 10 and matched_window(30) == 30


# ---------------------------------------------------------------------------
# 성과 · 백테스트
# ---------------------------------------------------------------------------

def test_paper_metrics_trades_and_sharpe():
    pred = np.array([1, -1, 0, 1])
    label = np.array([1, 1, 0, -1])
    r = np.array([0.02, 0.01, 0.0, -0.01])
    m = paper_metrics(pred, label, r, horizon=5)
    pnl = np.array([0.02, -0.01, -0.01])
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["n_trades"] == 3
    assert m["win_rate"] == pytest.approx(1 / 3)
    assert m["ret_per_trade"] == pytest.approx(pnl.mean())
    assert m["sharpe"] == pytest.approx(np.sqrt(252 / 5) * pnl.mean() / pnl.std(ddof=1))


def test_tranche_backtest_ramps_up_and_respects_delay():
    idx = pd.bdate_range("2020-01-01", periods=12)
    close = pd.Series(100 * 1.01 ** np.arange(12), index=idx)
    sig = pd.Series(1, index=idx)
    bt = tranche_backtest(sig, close, horizon=4, delay=0, cost_bps=0.0)
    np.testing.assert_allclose(bt["weight"].iloc[:5], [0.25, 0.5, 0.75, 1.0, 1.0])
    np.testing.assert_allclose(bt["strat_ret"], bt["weight"] * 0.01)
    late = tranche_backtest(sig, close, horizon=4, delay=2, cost_bps=0.0)
    np.testing.assert_allclose(late["weight"].iloc[:4], [0.0, 0.0, 0.25, 0.5])
    costly = tranche_backtest(sig, close, horizon=4, delay=0, cost_bps=10.0)
    np.testing.assert_allclose((bt["strat_ret"] - costly["strat_ret"]).iloc[:5],
                               [0.25e-3, 0.25e-3, 0.25e-3, 0.25e-3, 0.0])


def test_tranche_backtest_long_only_ignores_short_signals():
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(np.linspace(100, 90, 10), index=idx)
    sig = pd.Series([-1, 1] * 5, index=idx)
    ls = tranche_backtest(sig, close, horizon=1, delay=0, cost_bps=0.0)
    lo = tranche_backtest(sig, close, horizon=1, delay=0, cost_bps=0.0, long_only=True)
    assert set(ls["weight"].unique()) == {-1.0, 1.0}
    assert set(lo["weight"].unique()) == {0.0, 1.0}


# ---------------------------------------------------------------------------
# 모델 · 전체 실행
# ---------------------------------------------------------------------------

def _needs(*modules):
    return [pytest.mark.skipif(any(__import__("importlib").util.find_spec(m) is None for m in modules),
                               reason=f"{modules} 없음")]


@pytest.mark.parametrize("model,kwargs", [
    ("knn", {}), ("svm", {"gammas": (0.1,), "cs": (1.0,)}), ("ann", {"seeds": (0,), "max_epochs": 300}),
    pytest.param("ann", {"seeds": (0, 1), "max_epochs": 300, "backend": "torch"}, id="ann-torch",
                 marks=_needs("torch")),
    ("rf", {"depths": (3,), "leaves": (5,), "n_estimators": 50}), ("logit", {}),
    pytest.param("xgb", {"depths": (2,), "lrs": (0.1,), "trees": (20, 50)}, marks=_needs("xgboost")),
    pytest.param("lgbm", {"depths": (2,), "lrs": (0.1,), "trees": (20, 50)}, marks=_needs("lightgbm"))])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_models_learn_a_separable_rule(model, kwargs, n_classes):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = np.where(X[:, 0] > 0, 1, -1) if n_classes == 2 else np.digitize(X[:, 0], [-0.43, 0.43]) - 1
    pred, _ = fit_predict(model, X[:300], y[:300], X[300:], gap=2, **kwargs)
    assert set(np.unique(pred)) <= set(np.unique(y))
    assert np.mean(pred == y[300:]) > (0.85 if n_classes == 2 else 0.7)


def test_svm_direct_kernel_path_matches_precomputed():
    """cuML backend 가 쓰는 '커널을 SVC 가 직접 계산' 경로가 sklearn precomputed 경로와 같은 답을 낸다."""
    from sklearn.svm import SVC
    rng = np.random.default_rng(3)
    X = rng.normal(size=(240, 4))
    y = np.where(X[:, 0] + 0.5 * rng.normal(size=240) > 0, 1, -1)
    grid = {"gammas": (0.05, 0.2), "cs": (0.5, 4.0)}
    ref, ref_info = fit_predict_svm(X[:180], y[:180], X[180:], gap=2, **grid)

    def make(gamma, c):
        return SVC(kernel="sigmoid", gamma=gamma, coef0=0.0, C=c, max_iter=1_000_000)
    pred, info = _svm_direct(make, X[:180], y[:180], X[180:], 2, 5, grid["gammas"], grid["cs"])
    assert (info["gamma"], info["C"]) == (ref_info["gamma"], ref_info["C"])
    np.testing.assert_array_equal(pred, ref)


def simulate_arma(n=2500, seed=0, mu=3e-4, phi=0.3, theta=0.2, sd=0.01):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sd, n + 1)
    z = np.zeros(n + 1)
    for t in range(1, n + 1):
        z[t] = phi * z[t - 1] + e[t] + theta * e[t - 1]
    return mu + z[1:]


@pytest.mark.skipif(__import__("importlib").util.find_spec("statsmodels") is None, reason="statsmodels 없음")
def test_arma_forecast_matches_statsmodels_and_uses_only_past():
    from statsmodels.tsa.arima.model import ARIMA
    r = simulate_arma()
    p, q, mu, ar, ma, _ = select_arma(r, orders=((1, 1),))
    res = ARIMA(r, order=(1, 0, 1), trend="c").fit()
    for h in (1, 5, 20):
        ours = arma_cum_forecast(r, mu, ar, ma, np.array([len(r) - 1]), h)[0]
        assert ours == pytest.approx(res.forecast(h).sum(), rel=1e-4, abs=1e-9)
    origins = np.arange(1000, 1500)
    base = arma_cum_forecast(r, mu, ar, ma, origins, 5)
    shocked = r.copy()
    shocked[1300:] += 0.05
    after = arma_cum_forecast(shocked, mu, ar, ma, origins, 5)
    np.testing.assert_array_equal(base[origins < 1300], after[origins < 1300])
    assert not np.allclose(base[origins >= 1300], after[origins >= 1300])


@pytest.mark.skipif(__import__("importlib").util.find_spec("statsmodels") is None, reason="statsmodels 없음")
def test_arima_predicts_direction_of_autocorrelated_returns():
    r = simulate_arma(n=3000, mu=0.0, phi=0.6, theta=0.0)
    s, fit = 1, 2000
    pred, info = fit_predict("arima", r[:fit, None], np.zeros(fit), r[fit:-1, None], gap=s - 1, horizon=s,
                             delta=0.0, n_classes=2, orders=((0, 0), (1, 0), (2, 0)))
    assert info["p"] >= 1
    actual = np.where(r[fit + 1:] > 0, 1, -1)
    assert np.mean(pred == actual) > 0.65


def test_price_inputs_split_at_cutoff():
    df = simulate_ohlc()
    p = prepare(df, Scenario("close_long", True, df.index[0]), (3, 5), (1, 5), warmup=40)
    cutoff, te = p.dates[500], p.dates[500:700]
    X_tr, X_te, kw = price_inputs(p, p.dates[0], cutoff, te)
    lr = np.log(p.close).diff()
    np.testing.assert_allclose(X_tr[:, 0], lr.loc[p.dates[0]:p.dates[499]])
    np.testing.assert_allclose(X_te[kw["positions"], 0], lr.loc[te])
    assert len(X_te) == 200 and kw["positions"][0] == 0


def test_grid_runs_window_free_models_once_and_cache_names():
    df = simulate_ohlc(n=700, ohlc_from=0)
    p = prepare(df, Scenario("close_long", True, df.index[0]), (3, 5, 7), (1, 5), warmup=40)
    test = p.dates[-150:]
    jobs, _ = grid_jobs([p], test, ("knn", "arima"), (1, 5), (3, 5, 7), (2,))
    arima = [j for j in jobs if j.model == "arima"]
    assert len(arima) == 2 * 3 and len({j.key for j in arima}) == 2
    assert {j.meta["window"] for j in arima} == {3, 5, 7}
    assert cache_suffix({"backend": "sklearn", "max_iter": 10}) == ""
    assert cache_suffix({"backend": "torch", "device": "cuda"}) == "_torch-cuda"
    assert cache_suffix({"device": "cpu"}) == ""


def test_run_jobs_dedupes_keys():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = np.where(X[:, 0] > 0, 1, -1)
    job = Job("same", "knn", X[:100], y[:100], X[100:], 0, {})
    out = run_jobs([job, Job("same", "knn", X[:100], y[:100], X[100:], 0, {}), Job("other", "logit", X[:100], y[:100],
                                                                                    X[100:], 0, {})])
    assert set(out) == {"same", "other"}


def test_check_dependencies_reports_missing_package(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "xgboost" else real(name, *a))
    with pytest.raises(ValueError, match="pip install xgboost"):
        check_dependencies(("knn", "xgb"), {})
    check_dependencies(("knn", "rf", "logit"), {"svm": {"backend": "sklearn"}})
    monkeypatch.setattr("ti_forecast.models.gpu_available", lambda: False)
    with pytest.raises(ValueError, match="GPU"):
        check_dependencies(("knn",), {"knn": {"device": "cuda"}})


def test_cli_end_to_end(tmp_path):
    df = simulate_ohlc(n=1300, ohlc_from=250)
    path = tmp_path / "px.csv"
    df.reset_index().to_csv(path, index=False)
    out = tmp_path / "out"
    rc = run_ti_forecast.main([str(path), "--out", str(out), "--models", "knn", "--horizons", "1,5",
                               "--windows", "3,5", "--warmup", "40", "--wf-min-train-years", "1", "--n-jobs", "1",
                               "-q"])
    assert rc == 0
    for name in ("summary.md", "grid_metrics.csv", "grid_pattern.csv", "grid_portfolio.csv", "wf_metrics.csv",
                 "wf_horizon_blend.csv", "grid_heatmap_close_long_c3_sharpe.png",
                 "wf_equity_close_long_c2_long_only_full.png"):
        assert (out / name).exists(), name
    metrics = pd.read_csv(out / "grid_metrics.csv")
    assert set(metrics["scenario"]) == {"ohlc_short", "close_short", "close_long"}
    assert len(metrics[metrics["model"] == "knn"]) == 3 * 2 * 2 * 2
    # 같은 평가구간이므로 Buy & Hold 지표는 시나리오끼리 같다 (2 클래스)
    bh = metrics[(metrics["model"] == "buy_hold") & (metrics["n_classes"] == 2)]
    assert bh.groupby("horizon")["accuracy"].nunique().max() == 1
    # 계량 모형도 같은 경로로 돈다 (ARIMA 는 윈도우마다 같은 값)
    rc = run_ti_forecast.main([str(path), "--out", str(out / "more"), "--models", "logit,arima", "--horizons", "1,5",
                               "--windows", "3,5", "--warmup", "40", "--wf-min-train-years", "1", "--n-jobs", "2",
                               "--scenarios", "close_long", "-q"])
    assert rc == 0
    more = pd.read_csv(out / "more" / "grid_metrics.csv")
    assert set(more["model"]) == {"logit", "arima", "buy_hold"}
    assert more[more["model"] == "arima"].groupby(["n_classes", "horizon"])["accuracy"].nunique().max() == 1
    assert set(pd.read_csv(out / "more" / "wf_metrics.csv")["model"]) == {"logit", "arima", "buy_hold"}
    # 두 번째 실행은 캐시를 쓴다
    assert run_ti_forecast.main([str(path), "--out", str(out), "--models", "knn", "--horizons", "1,5",
                                 "--windows", "3,5", "--warmup", "40", "--wf-min-train-years", "1",
                                 "--n-jobs", "1", "-q", "--skip-wf"]) == 0

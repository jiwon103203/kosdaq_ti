#!/usr/bin/env python
"""기술지표 입력 윈도우 × 예측기간 방향 예측 실험 (Shynkevich et al. 2017) — 진입점.

    python run_ti_forecast.py data.xlsx                         # 그리드(논문 Table III–X) + walk-forward
    python run_ti_forecast.py data.xlsx --skip-wf --models svm  # 그리드만, SVM 만
    python run_ti_forecast.py data.xlsx --scenarios close_long --skip-grid
    python run_ti_forecast.py data.xlsx --models rf,xgb,lgbm,logit,arima   # 트리 계열 · 계량 모형
    python run_ti_forecast.py data.xlsx --models all --device cuda --ann-backend torch   # GPU

결과 (``--out``, 기본 out/ti_forecast):
    summary.md                 데이터 · 분할 · 핵심 표
    grid_metrics.csv           설정별 논문 지표 (정확도 · 승률 · 거래당 수익률 · Sharpe) + Buy & Hold
    grid_best_by_horizon.csv   논문 Table X (예측기간마다 윈도우 중 최고값)
    grid_pattern.csv           'n ≈ s 가 최고' 패턴 점검
    grid_portfolio.csv         tranche 포트폴리오 성과 (지연 · 거래비용 포함)
    grid_heatmap_*.png         예측기간 × 윈도우 히트맵
    wf_metrics*.csv, wf_portfolio*.csv, wf_equity_*.png         walk-forward (n = s, 매년 재학습)
    grid_predictions.csv.gz, wf_predictions.csv.gz              일별 예측
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from typing import Optional, Sequence

import pandas as pd

from ti_forecast.data import load_ohlc
from ti_forecast.experiment import (HORIZONS, SCENARIOS, WINDOWS, best_by_horizon, build_scenarios, common_test_dates,
                                    config_hash, grid_jobs, grid_metrics, grid_predictions, matched_window,
                                    performance_metrics, portfolio_table, prepare, run_jobs, tranche_backtest,
                                    walk_forward_jobs, walk_forward_metrics, walk_forward_predictions, window_pattern)
from ti_forecast.models import MODELS, PAPER_MODELS, check_dependencies

logger = logging.getLogger("run_ti_forecast")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
PORT_COLS = ["cagr", "ann_vol", "sharpe", "max_drawdown", "avg_weight", "turnover"]


def _ints(text: str):
    return tuple(int(x) for x in text.split(",") if x.strip())


def _names(choices):
    def parse(text: str):
        if text.strip() == "all":
            return tuple(choices)
        names = tuple(x.strip() for x in text.split(",") if x.strip())
        bad = [n for n in names if n not in choices]
        if bad:
            raise argparse.ArgumentTypeError(f"{bad} 는 없는 이름입니다. 가능: {list(choices)}")
        return names
    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="기술지표 입력 윈도우 × 예측기간 방향 예측 (Shynkevich et al. 2017)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="시가·고가·저가·종가 파일 (DataGuide xlsx 또는 csv)")
    p.add_argument("--sheet", default=None, help="엑셀 시트 (기본: 첫 시트)")
    p.add_argument("--out", "--out-dir", dest="out_dir", default=os.path.join("out", "ti_forecast"), help="출력 폴더")
    p.add_argument("--start", help="데이터 사용 시작일")
    p.add_argument("--end", help="데이터 사용 종료일")

    g = p.add_argument_group("실험")
    g.add_argument("--scenarios", type=_names(SCENARIOS), default=SCENARIOS,
                   help="ohlc_short = OHLC 지표 · close_short = 같은 기간 종가 전용 · close_long = 종가 전 기간")
    g.add_argument("--models", type=_names(MODELS), default=PAPER_MODELS,
                   help="논문 svm,ann,knn · 트리 rf,xgb,lgbm · 계량 logit,arima (쉼표로 여러 개, all = 전부)")
    g.add_argument("--horizons", type=_ints, default=HORIZONS, help="예측기간 s (거래일)")
    g.add_argument("--windows", type=_ints, default=WINDOWS, help="입력 윈도우 n (거래일)")
    g.add_argument("--classes", type=_ints, default=(2, 3), help="2 = Up/Down, 3 = Up/No Move/Down")
    g.add_argument("--warmup", type=int, default=120, help="지표 계산 후 버리는 앞쪽 거래일 (EWM 초기값 영향)")
    g.add_argument("--test-frac", type=float, default=1 / 3, help="그리드 평가구간 비율 (논문 2:1)")
    g.add_argument("--skip-grid", action="store_true", help="그리드 실험을 건너뜀")
    g.add_argument("--skip-wf", action="store_true", help="walk-forward 를 건너뜀")
    g.add_argument("--wf-min-train-years", type=int, default=3,
                   help="walk-forward 첫 재학습 연도 = 표본 시작 연도 + 이 값 (하반기 시작이면 +1 덜 함)")
    g.add_argument("--wf-train-years", type=int, default=None, help="walk-forward 학습창 (기본: 처음부터 전부)")

    g = p.add_argument_group("모델")
    g.add_argument("--ann-seeds", type=int, default=5, help="ANN 시드 수 (확률 평균)")
    g.add_argument("--svm-max-iter", type=int, default=1_000_000, help="SVM solver 최대 반복")
    g.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1, help="병렬 작업 수 (CPU 작업)")

    g = p.add_argument_group("가속 (GPU)")
    g.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="cuda = xgb 와 --ann-backend torch 를 GPU 에서 학습")
    g.add_argument("--ann-backend", choices=("sklearn", "torch"), default="sklearn",
                   help="torch = 시드 전부를 한 번에 학습 (CPU 에서도 빠름, 결과 숫자는 sklearn 과 조금 다름)")
    g.add_argument("--svm-backend", choices=("sklearn", "cuml"), default="sklearn",
                   help="cuml = RAPIDS cuML GPU SVC 로 같은 격자 탐색")
    g.add_argument("--gpu-jobs", type=int, default=1, help="GPU 작업을 동시에 돌릴 프로세스 수")
    g.add_argument("--no-cache", action="store_true", help="끝난 학습 결과를 캐시하지 않음")

    g = p.add_argument_group("포트폴리오 백테스트")
    g.add_argument("--delay", type=int, default=1, help="신호 다음 몇 거래일 종가에 진입 (0 = 논문처럼 당일 종가)")
    g.add_argument("--cost-bps", type=float, default=10.0, help="비중 변화 1 당 거래비용 (bp)")

    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def first_refit_year(dates: pd.DatetimeIndex, min_years: int) -> int:
    return dates[0].year + min_years + (1 if dates[0].month > 6 else 0)


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """tabulate 없이 마크다운 표."""
    df = df.reset_index()
    cells = [[floatfmt.format(v) if isinstance(v, float) else str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def horizon_blend(pred: pd.DataFrame, close: pd.Series, delay: int, cost_bps: float,
                  long_only: bool) -> pd.DataFrame:
    """예측기간별 tranche 전략의 동일가중 평균 (예측기간을 고르지 않은 요약). 열: ``strat_ret``, ``weight``.

    거래비용은 예측기간 전략마다 따로 뺀 뒤 평균하므로, 전략끼리 상쇄되는 매매까지 비용을 낸 보수적인 값이다.
    """
    bts = []
    for s, g in pred.groupby("horizon"):
        sig = pd.Series(g["pred"].to_numpy(), index=pd.DatetimeIndex(g["date"])).sort_index()
        bts.append(tranche_backtest(sig, close, int(s), delay, cost_bps, long_only))
    return pd.DataFrame({c: pd.concat([bt[c] for bt in bts], axis=1).mean(axis=1) for c in ("strat_ret", "weight")})


def run(args: argparse.Namespace) -> dict:
    from ti_forecast.plotting import MODEL_LABEL, equity_curves, grid_heatmaps

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_ohlc(args.input, args.sheet).loc[args.start:args.end]
    scenarios = build_scenarios(df, args.scenarios)
    prepared = {sc.name: prepare(df, sc, args.windows, args.horizons, args.warmup) for sc in scenarios}
    model_kwargs = {"svm": {"max_iter": args.svm_max_iter}, "ann": {"seeds": tuple(range(args.ann_seeds))}}
    # backend · device 는 캐시 폴더가 아니라 파일 이름에 붙는다 (cache_suffix) → 기본 설정이면 예전 캐시를 그대로 쓴다
    runtime = {"svm": {"backend": args.svm_backend}, "xgb": {"device": args.device},
               "ann": {"backend": args.ann_backend, **({"device": args.device} if args.ann_backend == "torch" else {})}}
    fit_kwargs = {m: {**model_kwargs.get(m, {}), **runtime.get(m, {})} for m in MODELS}
    check_dependencies(args.models, fit_kwargs)
    if args.device == "cuda" and not any(m in args.models and fit_kwargs[m].get("device") == "cuda"
                                         for m in ("xgb", "ann")):
        logger.warning("--device cuda 는 xgb 와 --ann-backend torch 에만 쓰입니다 (지금 설정에서는 효과 없음)")
    cache_cfg = {"data": _file_sha1(args.input), "sheet": args.sheet, "start": args.start, "end": args.end,
                 "warmup": args.warmup, "test_frac": args.test_frac, "wf_min_train_years": args.wf_min_train_years,
                 "wf_train_years": args.wf_train_years, "model_kwargs": model_kwargs}
    cache_dir = None if args.no_cache else os.path.join(args.out_dir, "cache", config_hash(cache_cfg))
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "cache_dir": cache_dir}, f, ensure_ascii=False, indent=2, default=str)

    md = [f"# 기술지표 윈도우 × 예측기간 실험\n\n입력: `{args.input}` · 진입 지연 {args.delay}일 · "
          f"거래비용 {args.cost_bps:g}bp\n\n## 시나리오 표본\n"]
    md.append(_md(pd.DataFrame([{"scenario": n, "close_only": p.scenario.close_only, "first": p.dates[0].date(),
                                 "last": p.dates[-1].date(), "n_days": len(p.dates)}
                                for n, p in prepared.items()]).set_index("scenario")))
    out = {}

    if not args.skip_grid:
        test = common_test_dates(list(prepared.values()), args.horizons, args.test_frac)
        logger.info("grid test period %s ~ %s (%d days)", test[0].date(), test[-1].date(), len(test))
        jobs, thresholds = grid_jobs(list(prepared.values()), test, args.models, args.horizons, args.windows,
                                     args.classes)
        results = run_jobs(jobs, args.n_jobs, fit_kwargs, cache_dir, "grid", args.gpu_jobs)
        metrics = grid_metrics(jobs, results, test)
        preds = grid_predictions(jobs, results, test)
        pattern = window_pattern(metrics)
        port = portfolio_table(preds, df["close"], args.delay, args.cost_bps)
        thr = pd.DataFrame(thresholds).T.rename_axis(["scenario", "horizon"])
        metrics.to_csv(os.path.join(args.out_dir, "grid_metrics.csv"), index=False)
        preds.to_csv(os.path.join(args.out_dir, "grid_predictions.csv.gz"), index=False)
        best_by_horizon(metrics).to_csv(os.path.join(args.out_dir, "grid_best_by_horizon.csv"))
        pattern.to_csv(os.path.join(args.out_dir, "grid_pattern.csv"), index=False)
        port.to_csv(os.path.join(args.out_dir, "grid_portfolio.csv"), index=False)
        thr.to_csv(os.path.join(args.out_dir, "grid_thresholds.csv"))
        for name in prepared:
            for k in args.classes:
                for metric in ("accuracy", "win_rate", "ret_per_trade", "sharpe"):
                    if metric == "win_rate" and k == 2:
                        continue  # 2 클래스에서는 정확도와 같다
                    grid_heatmaps(metrics, name, k, metric,
                                  os.path.join(args.out_dir, f"grid_heatmap_{name}_c{k}_{metric}.png"), args.models)

        on_diag = metrics[(metrics["model"] == "buy_hold") | (metrics["window"] == metrics["horizon"].map(
            lambda s: matched_window(int(s), args.windows)))]
        diag = {m: on_diag.pivot_table(index=["scenario", "n_classes", "model"], columns="horizon", values=m)
                for m in ("accuracy", "sharpe")}
        md += [f"\n\n## 그리드 (평가 {test[0].date()} ~ {test[-1].date()}, {len(test)}일)\n",
               "\n### 3 클래스 No Move 기준값 δ (학습구간 |r| 1/3 분위수)\n",
               _md(thr["delta"].unstack("horizon").astype(float), "{:.4f}"),
               "\n\n### 'n ≈ s 가 최고' 패턴 (정확도 · Sharpe)\n",
               _md(pattern[pattern["metric"].isin(["accuracy", "sharpe"])].set_index(
                   ["scenario", "model", "n_classes", "metric"]), "{:.3f}"),
               "\n\n### n ≈ s 칸의 정확도 (열 = 예측기간 s, buy_hold = 항상 Up)\n", _md(diag["accuracy"], "{:.3f}"),
               "\n\n### n ≈ s 칸의 Sharpe (논문 식 20)\n", _md(diag["sharpe"], "{:.2f}")]
        out.update(grid_metrics=metrics, grid_pattern=pattern, grid_portfolio=port)

    if not args.skip_wf:
        firsts = {n: first_refit_year(p.dates, args.wf_min_train_years) for n, p in prepared.items()}
        jobs = [j for n, p in prepared.items()
                for j in walk_forward_jobs(p, args.models, args.horizons, args.classes, firsts[n], args.windows,
                                           args.wf_train_years)]
        results = run_jobs(jobs, args.n_jobs, fit_kwargs, cache_dir, "walk-forward", args.gpu_jobs)
        wf = walk_forward_predictions(jobs, results, prepared)
        wf.to_csv(os.path.join(args.out_dir, "wf_predictions.csv.gz"), index=False)
        common = pd.Timestamp(f"{max(firsts.values())}-01-01")
        for suffix, start in (("", None), ("_common", common)):
            sub = wf if start is None else wf[wf["date"] >= start]
            wm = walk_forward_metrics(sub)
            wp = portfolio_table(sub, df["close"], args.delay, args.cost_bps)
            wm.to_csv(os.path.join(args.out_dir, f"wf_metrics{suffix}.csv"), index=False)
            wp.to_csv(os.path.join(args.out_dir, f"wf_portfolio{suffix}.csv"), index=False)
            out[f"wf_metrics{suffix}"], out[f"wf_portfolio{suffix}"] = wm, wp

        blend_rows, hs = [], ",".join(map(str, args.horizons))
        for (name, k), g in wf.groupby(["scenario", "n_classes"]):
            for start, tag in ((None, "full"), (common, "common")):
                gg = g if start is None else g[g["date"] >= start]
                dates = pd.DatetimeIndex(sorted(gg["date"].unique()))
                bench = df["close"].reindex(dates).pct_change().iloc[1:]
                blend_rows.append({"scenario": name, "n_classes": k, "period": tag, "strategy": "buy_hold",
                                   **performance_metrics(bench, weight=pd.Series(1.0, index=bench.index))})
                for long_only in (False, True):
                    mode = "long_only" if long_only else "long_short"
                    curves = {}
                    for m, gm in gg.groupby("model"):
                        bt = horizon_blend(gm, df["close"], args.delay, args.cost_bps, long_only)
                        curves[MODEL_LABEL.get(m, m)] = bt["strat_ret"]
                        blend_rows.append({"scenario": name, "n_classes": k, "period": tag, "strategy": f"{m}_{mode}",
                                           **performance_metrics(bt["strat_ret"], weight=bt["weight"])})
                    equity_curves(curves, bench,
                                  f"Walk-forward {name}, {k}-class, {mode}, n = s, horizons {hs} equally weighted "
                                  f"({dates[0].date()} ~ {dates[-1].date()})",
                                  os.path.join(args.out_dir, f"wf_equity_{name}_c{k}_{mode}_{tag}.png"))
        blend = pd.DataFrame(blend_rows)
        blend.to_csv(os.path.join(args.out_dir, "wf_horizon_blend.csv"), index=False)
        out["wf_horizon_blend"] = blend
        md += ["\n\n## walk-forward (n = s, 매년 재학습)\n",
               f"첫 재학습 연도: {firsts}. 공통 비교 구간: {common.date()} 이후.\n",
               f"\n### 예측기간 {hs} 동일가중 tranche 포트폴리오\n",
               _md(blend.set_index(["scenario", "n_classes", "period", "strategy"])[PORT_COLS], "{:.3f}")]

    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    try:
        run(args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 1
    print(f"결과: {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

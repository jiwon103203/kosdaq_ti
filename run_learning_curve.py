#!/usr/bin/env python
"""학습 곡선 — 학습 기간(년)을 바꿔 가며 같은 평가구간에서 walk-forward 성과를 비교한다.

    python run_learning_curve.py data.xlsx                                  # 일간, 학습 1,2,3,5,7,10년
    python run_learning_curve.py data.xlsx --freq W                         # 일간 입력을 주간으로 바꿔서
    python run_learning_curve.py data.xlsx --train-years 2,4,6,8 --models logit,knn,ann,svm
    python run_learning_curve.py data.xlsx --horizons 5,10,20 --classes 2   # 주 단위 예측을 일간 데이터로

매년 첫 표본 날짜에 '최근 k 년'(rolling)으로 다시 학습해 그 해를 예측한다(입력 윈도우 n = s). 평가 첫해는 가장 긴 학습
기간이 표본 안에 들어가는 해(``--eval-start-year`` 로 바꿀 수 있음)이므로, 학습 기간마다 평가일이 같다.

결과 (``--out``, 기본 out/learning_curve):
    summary.md                 표본 · 평가구간, 학습 기간별 Buy & Hold 대비 정확도 차이, CV 부풀림, 필요한 표본 수
    lc_metrics.csv             (학습 기간, 모델, 클래스 수, 예측기간)별 정확도 · edge · mde · cv_acc · 학습 표본 수
    lc_summary.csv             예측기간 평균
    lc_sample_size.csv         엣지 d 를 확인하는 데 필요한 독립 표본 수 · 평가 기간(년)
    learning_curve_c*.png      학습 기간 × Buy & Hold 대비 정확도 차이 (회색 띠 = 구분할 수 없는 범위)
    lc_predictions.csv.gz      기간별 예측
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional, Sequence

from run_ti_forecast import LOG_FORMAT, _file_sha1, _ints, _md, _names, load_prices
from ti_forecast.experiment import SCENARIOS, build_scenarios, config_hash, prepare, run_jobs
from ti_forecast.learning_curve import (default_first_year, learning_curve_jobs, learning_curve_metrics,
                                        learning_curve_predictions, learning_curve_summary, sample_size_table)
from ti_forecast.models import MODELS, check_dependencies

logger = logging.getLogger("run_learning_curve")
DEFAULT_MODELS = ("logit", "knn", "ann")  # SVM 은 1회 학습이 길어(약 100초) 기본에서 뺐다. --models 로 추가
EDGES = (0.01, 0.02, 0.03, 0.05)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="학습 기간별 walk-forward 학습 곡선",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="시가·고가·저가·종가 파일 (DataGuide xlsx 또는 csv)")
    p.add_argument("--sheet", default=None, help="엑셀 시트 (기본: 첫 시트)")
    p.add_argument("--out", "--out-dir", dest="out_dir", default=os.path.join("out", "learning_curve"),
                   help="출력 폴더")
    p.add_argument("--start", help="데이터 사용 시작일")
    p.add_argument("--end", help="데이터 사용 종료일")
    p.add_argument("--freq", choices=("auto", "D", "W"), default="auto",
                   help="auto = 입력 날짜 간격으로 판별, D = 일간, W = 주간 (일간 입력이면 주간으로 바꿈)")

    g = p.add_argument_group("실험")
    g.add_argument("--scenario", choices=SCENARIOS, default="close_long",
                   help="표본이 가장 긴 close_long 이 기본 (OHLC 시나리오는 학습 기간을 길게 잡을 수 없다)")
    g.add_argument("--train-years", type=_ints, default=(1, 2, 3, 5, 7, 10), help="비교할 학습 기간 (년, rolling)")
    g.add_argument("--eval-start-year", type=int, default=None,
                   help="평가 첫해 (기본: 가장 긴 학습 기간이 표본 안에 들어가는 첫해)")
    g.add_argument("--models", type=_names(MODELS), default=DEFAULT_MODELS,
                   help="svm,ann,knn,rf,xgb,lgbm,logit,arima 중 쉼표로 (all = 전부)")
    g.add_argument("--horizons", type=_ints, default=None,
                   help="예측기간 s (기간 단위. 기본: 일간 1,3,5,7,10,15,20,25,30 · 주간 1,2,3,4,6,8,13)")
    g.add_argument("--windows", type=_ints, default=None, help="입력 윈도우 후보 (n = s 에 가장 가까운 것을 씀)")
    g.add_argument("--classes", type=_ints, default=(2, 3), help="2 = Up/Down, 3 = Up/No Move/Down")
    g.add_argument("--warmup", type=int, default=None, help="지표 계산 후 버리는 앞쪽 기간 수 (기본: 일간 120 · 주간 60)")

    g = p.add_argument_group("모델")
    g.add_argument("--ann-seeds", type=int, default=5, help="ANN 시드 수 (확률 평균)")
    g.add_argument("--svm-max-iter", type=int, default=1_000_000, help="SVM solver 최대 반복")
    g.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1, help="병렬 작업 수")
    g.add_argument("--no-cache", action="store_true", help="끝난 학습 결과를 캐시하지 않음")

    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict:
    from ti_forecast.plotting import MODEL_LABEL, learning_curves

    os.makedirs(args.out_dir, exist_ok=True)
    df, freq, spec = load_prices(args)
    periods, unit = spec.periods_per_year, spec.unit
    sc = build_scenarios(df, [args.scenario])[0]
    prep = prepare(df, sc, args.windows, args.horizons, args.warmup)
    train_years = tuple(sorted(set(args.train_years)))
    if not train_years or min(train_years) < 1:
        raise ValueError("--train-years 는 1 이상이어야 합니다")
    first = args.eval_start_year or default_first_year(prep.dates, max(train_years))
    if first is None or first > prep.dates[-1].year:
        span = (prep.dates[-1] - prep.dates[0]).days / 365.25
        raise ValueError(f"표본이 {span:.1f}년이라 학습 {max(train_years)}년 + 평가 1년이 들어가지 않습니다. "
                         f"--train-years 를 줄이세요")
    model_kwargs = {"svm": {"max_iter": args.svm_max_iter}, "ann": {"seeds": tuple(range(args.ann_seeds))}}
    check_dependencies(args.models, model_kwargs)
    cache_cfg = {"tool": "learning_curve", "data": _file_sha1(args.input), "sheet": args.sheet, "start": args.start,
                 "end": args.end, "freq": freq, "warmup": args.warmup, "scenario": args.scenario,
                 "model_kwargs": model_kwargs}
    cache_dir = None if args.no_cache else os.path.join(args.out_dir, "cache", config_hash(cache_cfg))
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "freq": freq, "eval_start_year": first, "cache_dir": cache_dir}, f,
                  ensure_ascii=False, indent=2, default=str)

    jobs = learning_curve_jobs(prep, args.models, args.horizons, args.classes, first, args.windows, train_years)
    logger.info("learning curve: train years %s, evaluation from %d, %d jobs", train_years, first, len(jobs))
    results = run_jobs(jobs, args.n_jobs, model_kwargs, cache_dir, "learning-curve")
    pred = learning_curve_predictions(jobs, results, {sc.name: prep})
    metrics = learning_curve_metrics(pred, jobs, results, periods)
    summary = learning_curve_summary(metrics)
    sizes = sample_size_table(EDGES, args.horizons, periods)
    pred.to_csv(os.path.join(args.out_dir, "lc_predictions.csv.gz"), index=False)
    metrics.to_csv(os.path.join(args.out_dir, "lc_metrics.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "lc_summary.csv"), index=False)
    sizes.to_csv(os.path.join(args.out_dir, "lc_sample_size.csv"), index=False)
    for k in args.classes:
        learning_curves(metrics, k, os.path.join(args.out_dir, f"learning_curve_c{k}.png"), args.models, spec.unit_en)

    ev = metrics.iloc[0]
    md = [f"# 학습 곡선 (학습 기간별 walk-forward)\n\n입력: `{args.input}` · {'주간' if freq == 'W' else '일간'} "
          f"(단위 = {unit}) · 시나리오 `{sc.name}` · 입력 윈도우 n = s · 매년 재학습, 최근 k 년만 학습(rolling)\n",
          f"\n- 표본: {prep.dates[0].date()} ~ {prep.dates[-1].date()} ({len(prep.dates)}{unit})",
          f"- 평가구간 (모든 학습 기간 공통): {ev['start']} ~ {metrics['end'].max()} "
          f"(약 {int(metrics['n_eval'].max())}{unit}, 예측기간마다 레이블이 확정된 날만)",
          f"- 학습 기간: {', '.join(map(str, train_years))}년 · 모델: {', '.join(MODEL_LABEL.get(m, m) for m in args.models)}",
          "\n`edge` = 정확도 − Buy & Hold(항상 Up) 정확도. `mde` = 이 평가구간으로 확인할 수 있는 최소 차이(α 5% 양측, "
          "검정력 80%, 독립 표본 N/s). `cv_gap` = 학습구간 교차검증 정확도 − 평가 정확도(하이퍼파라미터 선택 · 과적합으로 "
          "부푼 정도). `significant` = edge > mde 인 예측기간 비율. 3 클래스는 δ 를 학습구간에서 정하므로 학습 기간마다 "
          "평가 레이블과 Buy & Hold 정확도가 조금 다르다.\n"]
    for k in args.classes:
        sk = summary[summary["n_classes"] == k]
        md += [f"\n## {k} 클래스\n", "\n### Buy & Hold 대비 정확도 차이 (예측기간 평균, 행 = 학습 기간 · 열 = 모델)\n",
               _md(sk.pivot(index="train_years", columns="model", values="edge"), "{:+.3f}"),
               "\n\n### 예측기간 평균 요약\n",
               _md(sk.set_index(["model", "train_years"])[["accuracy", "bh_accuracy", "edge", "mde", "cv_acc",
                                                           "cv_gap", "n_train", "n_train_independent", "beats_bh",
                                                           "significant"]], "{:.3f}"),
               "\n\n### 예측기간별 edge (행 = 모델 · 학습 기간, 열 = 예측기간 s)\n",
               _md(metrics[metrics["n_classes"] == k].pivot_table(index=["model", "train_years"], columns="horizon",
                                                                  values="edge"), "{:+.3f}"),
               "\n\n최소 확인 가능 차이 mde (열 = 예측기간 s):\n",
               _md(metrics[metrics["n_classes"] == k].groupby("horizon")["mde"].first().to_frame().T
                   .rename_axis(index="metric"), "{:.3f}")]
    md += ["\n\n## 필요한 표본 수 (정확도 50% 근처, α 5% 양측, 검정력 80%)\n",
           f"\n엣지 d 를 확인하는 데 필요한 독립 표본 수와, 예측기간 s{unit} 레이블로 그만큼 쌓는 데 걸리는 기간(년, "
           f"1년 = {periods}{unit}). 학습 데이터도 적어도 이만큼은 있어야 그 크기의 패턴을 잡음과 구분해 배울 수 있다.\n",
           _md(sizes.set_index("edge_pp"), "{:.1f}"),
           "\n\n## 읽는 법\n",
           "\n- 곡선이 학습 기간에 따라 오르면 데이터가 더 있으면 나아질 여지가 있다. 평평하거나 회색 띠(잡음 범위) 안에 "
           "머물면 데이터를 늘리는 것만으로는 나아지지 않는다.",
           "- 짧은 학습 기간에서 `cv_gap` 이 크면 표본이 적어 교차검증으로 고른 하이퍼파라미터가 잡음에 맞춰진 것이다.",
           "- 긴 학습 기간이 오히려 나쁘면 시장 성격이 바뀌어(비정상성) 오래된 데이터가 방해가 되는 것일 수 있다."]
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return {"metrics": metrics, "summary": summary, "sample_size": sizes}


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

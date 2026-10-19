# kosdaq_ti

코스닥150 지수에 Shynkevich et al. (2017, *Neurocomputing*) **"Forecasting price movements using technical indicators: Investigating the impact of varying input window length"** 를 적용한 실험입니다.

논문은 10개 기술지표를 모두 같은 입력 윈도우 n으로 계산해 SVM · ANN · kNN으로 s일 뒤 가격 방향을 맞히고, 예측기간 s × 입력 윈도우 n 격자에서 **"n ≈ s일 때 성능이 가장 좋다"** 는 패턴을 보고합니다(S&P 500 종목 50개, 학습 1740일 · 평가 900일).

코스닥150은 종가가 2010-01-04부터, 시가·고가·저가가 2015-07-13부터만 있습니다. 그래서 고가·저가가 필요한 지표를 **종가만으로 계산하는 버전**(고가 = 저가 = 종가)을 만들어 학습 기간을 2010년까지 늘리고, OHLC 원식과 비교합니다.

## 실행

```bash
pip install -r requirements.txt
python run_ti_forecast.py data/kosdaq150.xlsx                       # 그리드(논문 Table III–X) + walk-forward
python run_ti_forecast.py data/kosdaq150.xlsx --models svm --skip-wf  # 그리드만, SVM만
python run_ti_forecast.py data/kosdaq150.xlsx --scenarios close_long --skip-grid --cost-bps 5
pytest tests                                                         # 지표 식 · 미래 정보 미사용 · 분할 규칙
```

입력은 DataGuide 엑셀(메타데이터 줄 뒤 `D A T E | 저가지수 | 고가지수 | 시가지수 | 종가지수`)이나 `date, open, high, low, close` csv입니다. 시가·고가·저가가 없는 날은 비워 두면 됩니다.

전체 실행(3 시나리오 × 9 예측기간 × 8 윈도우 × 2/3 클래스 × 3 모델 = 1,296개 학습 + 연도별 walk-forward)은 4코어에서 2~3시간 걸립니다. 대부분 SVM 격자 탐색(110개 조합 × 5 fold)이며, 끝난 학습은 `out/.../cache/`에 저장되어 다시 실행하면 재사용합니다.

## 기술지표 (모두 같은 입력 윈도우 n)

| 지표 | OHLC 원식 | 종가 전용 (H = L = C) |
|---|---|---|
| SMA | C / SMA_n(C) − 1 | 같음 |
| EMA | C / EMA_n(C) − 1 | 같음 |
| ATR | Wilder_n(TR) / C | Wilder_n(\|ΔC\|) / C |
| ADX | Wilder_n(DX), DX = 100\|DI⁺ − DI⁻\|/(DI⁺ + DI⁻) | DM = max(±ΔC, 0) |
| CCI | (TP − SMA_n TP) / (0.015 · 평균편차), TP = (H+L+C)/3 | TP = C |
| ROC | C_t / C_{t−n} − 1 | 같음 |
| RSI | 100 · 상승 / (상승 + 하락), Wilder 평활 | 같음 |
| Williams %R | −100 (HH_n − C) / (HH_n − LL_n) | HH, LL = n일 최고·최저 종가 |
| Stochastic %K | 100 (C − LL_n) / (HH_n − LL_n) | 〃 |
| Stochastic %D | EMA_3(%K) | 〃 |

종가 전용 버전은 따로 만든 지표가 아니라, 같은 식에 고가 = 저가 = 종가를 넣은 것입니다. SMA · EMA · ROC · RSI는 원래 종가만 씁니다. 평활은 논문이 쓴 TA-Lib처럼 ATR · RSI · DI · ADX는 Wilder(α = 1/n), EMA는 α = 2/(n+1)입니다.

## 실험 설계

**시나리오** — 셋을 같은 평가구간에서 비교해 지표 종류와 학습 기간의 효과를 나눠 봅니다.

| 시나리오 | 지표 | 학습 시작 | 비교 |
|---|---|---|---|
| `ohlc_short` | OHLC 원식 | 2015-07 (+ warmup 120일) | 논문 그대로 |
| `close_short` | 종가 전용 | `ohlc_short`와 같은 날짜 | 지표 종류만 바꾼 효과 |
| `close_long` | 종가 전용 | 2010-01 (+ warmup 120일) | 학습 기간을 늘린 효과 |

**그리드 (논문 Table III–X)** — 평가구간은 OHLC 표본의 마지막 1/3(논문 2:1, `--test-frac`)이고, 세 시나리오 모두 같은 평가일입니다. s ∈ {1, 3, 5, 7, 10, 15, 20, 25, 30} × n ∈ {3, 5, 7, 10, 15, 20, 25, 30} × 2 클래스(Up/Down) · 3 클래스(Up/No Move/Down) × 모델마다 한 번 학습해 평가구간 전체를 예측합니다. 3 클래스 기준값 δ는 학습구간 |s일 수익률|의 1/3 분위수입니다(논문 Table I과 같은 기준: 클래스마다 약 1/3).

**walk-forward** — 매년 첫 거래일에 그 전까지 레이블이 확정된 표본으로 다시 학습해 그 해를 예측합니다. 입력 윈도우는 논문의 결론대로 n = s(s < 3이면 3)로 고정하므로, 평가구간을 보고 윈도우를 고르는 일이 없습니다. 첫 재학습은 표본 시작 약 3년 뒤(`close_long` 2013년, OHLC 시나리오 2019년)이고, `*_common` 파일은 세 시나리오를 2019년 이후 같은 구간에서 비교합니다.

**모델 (논문 Sec. 5.5)**
- SVM: sigmoid 커널, γ ∈ {2⁻¹⁵, 2⁻¹³, …, 2³} × C ∈ {2⁻⁵, 2⁻³, …, 2¹⁵}를 5-fold 교차검증 정확도로 선택
- ANN: 입력 10 – 은닉 10(tanh) – softmax, 학습구간 뒤 25%로 early stopping, `--ann-seeds`(5)개 시드의 확률 평균
- kNN: k ∈ {1, …, 10}을 5-fold 교차검증 정확도로 선택

**성과**
- 논문 지표: 매 평가일 Up → s일 매수, Down → s일 매도, No Move → 거래 없음. 정확도, 승률, 거래당 수익률, Sharpe = √(252/s) · 평균 / 표준편차(무위험금리 0). 겹치는 s일 거래를 독립으로 세므로 낙관적입니다. `acc_se`는 겹침을 감안한 정확도 표준오차 근사 √(p(1−p)/(N/s))입니다.
- tranche 포트폴리오: 매일 신호의 1/s를 s일 보유하는 겹치는 포지션으로 일간 수익률을 만들고, 신호 다음 날 종가에 체결(`--delay 1`)하며, 비중 변화 1당 10bp(`--cost-bps`)를 뺍니다. 롱숏(논문)과 롱온리(Up일 때만 매수) 두 가지로 CAGR · Sharpe · MDD를 계산합니다.
- 비교 기준은 같은 기간 Buy & Hold입니다.

## 논문과 다르게 한 것

- **교차검증 fold**: 시간 순서 블록으로 나누고, 레이블 구간 (t, t+s]가 검증 블록과 겹치는 학습 표본을 버립니다(purged k-fold). 무작위 fold는 겹치는 레이블 때문에 검증 정확도가 부풀어 과적합된 하이퍼파라미터를 고릅니다. 학습/평가 경계에서도 레이블이 평가구간에 걸치는 마지막 s개 학습 표본을 버립니다.
- **SMA · EMA · ATR**: 가격 수준 대신 종가 대비 비율로 씁니다. 한 지수를 16년 쓰면 평가구간 가격이 학습구간 범위를 벗어나 표준화가 의미를 잃기 때문입니다(정보는 같습니다).
- **ANN**: Matlab trainscg 대신 Adam + 검증 손실 early stopping(patience 10)이고, 종목 50개 평균 대신 지수 하나라 초기값 운을 줄이려고 시드 5개를 평균합니다.
- **표본**: 종목 50개의 평균 ± 표준편차가 아니라 지수 하나라 칸마다 값이 하나이고 잡음이 큽니다(`acc_se` 참고).

## 결과 (코스닥150, `results/kosdaq150/`)

기본 설정(진입 1일 지연, 거래비용 10bp, 무위험금리 0)으로 돌린 결과입니다. 전체 표와 그림은 `results/kosdaq150/`에 있습니다(일별 예측 중 그리드 쪽은 15MB라 빼고, walk-forward 예측만 넣었습니다).

**그리드 (평가 2023-02-03 ~ 2026-08-31, 871거래일, 학습 OHLC/종가 1,740일 · 종가 2010~ 3,109일)**
- 정확도가 Buy & Hold(항상 Up)를 넘는 칸이 드뭅니다. 72칸 평균 정확도는 2 클래스에서 B&H보다 2.5~5%p, 3 클래스에서 5~9%p 낮습니다. B&H를 넘는 칸의 비율은 SVM 4~18%, ANN 21~31%, kNN 10% 안팎입니다.
- 평가구간에서 윈도우를 골라도(논문 Table X 방식, 낙관적) B&H를 확실히 넘는 것은 s = 1~5일의 일부뿐이고, s ≥ 15에서는 대부분 B&H보다 낮습니다.
- "n ≈ s가 최고" 패턴은 재현되지 않습니다. n ≈ s 칸의 윈도우 내 순위는 평균 3.5~5위/8(무작위면 4.5)입니다. 다만 종가 전용 SVM · ANN은 예측기간이 길수록 좋은 윈도우도 길어지는 경향(순위상관 0.4~0.9)이 약하게 보입니다.
- 정확도 표준오차(`acc_se`)가 약 5.5%p라 칸 사이 차이 대부분이 잡음 범위입니다.

**walk-forward (n = s, 매년 재학습, 거래비용 · 지연 포함)**
- 예측기간 1~30을 동일가중한 포트폴리오는 어느 모델도 B&H Sharpe를 넘지 못합니다.
  - `close_long`, 2013~2026: B&H Sharpe 0.25 · CAGR 3.1% · MDD −59%. 가장 나은 ANN 롱온리는 Sharpe 0.22 · CAGR 2.3% · MDD −44%.
  - 2019~2026 공통 구간: B&H 0.23. ANN 롱온리는 OHLC 0.22, 종가 전용(같은 기간) 0.22, 종가 전용(2010~ 학습) 0.18.
- 롱숏(논문 방식)은 대부분 Sharpe가 0 근처이거나 음수입니다.
- ANN 롱온리는 수익률은 B&H와 비슷하지만 평균 비중이 약 0.5~0.6이라, 변동성(약 20% vs 29~34%)과 MDD가 작습니다. 예측력보다는 비중을 줄인 효과에 가깝습니다.
- s = 1에서는 논문 지표 Sharpe가 B&H의 두 배(ANN · kNN 0.5 안팎 vs 0.25)이지만, 하루 지연과 거래비용을 넣은 포트폴리오에서는 대부분 사라집니다.

**종가 전용 vs OHLC, 학습 기간**
- 같은 날짜에서 종가 전용(`close_short`)과 OHLC(`ohlc_short`)는 그리드 · walk-forward 모두 차이가 잡음 수준입니다. 이 데이터에서는 고가 · 저가가 추가 정보를 거의 주지 않았습니다.
- 2010년부터 학습(`close_long`)하면 그리드에서 롱온리 성과가 조금 나아지지만(n ≈ s 칸 SVM Sharpe 중앙값 0.32 vs 0.08 · 0.16), 상당 부분은 모델이 Up을 더 자주 예측해 B&H에 가까워진 결과입니다. 그 대신 종가 전용으로 표본 밖 백테스트를 2013년부터 할 수 있어 검증 기간이 7.7년에서 13.8년으로 늘어납니다.

## 출력 (`--out`, 기본 `out/ti_forecast/`)

| 파일 | 내용 |
|---|---|
| `summary.md` | 시나리오 표본, δ, 패턴 점검, n ≈ s 칸 정확도 · Sharpe, walk-forward 요약 |
| `grid_metrics.csv` | 설정별 논문 지표 + 같은 평가구간 Buy & Hold(항상 Up) |
| `grid_best_by_horizon.csv` | 논문 Table X(예측기간마다 윈도우 중 최고값 — 평가구간에서 고른 값이라 낙관적) |
| `grid_pattern.csv` | 예측기간별 최고 윈도우, 예측기간과의 순위상관, n ≈ s 칸의 순위 · 평균 대비 차이 |
| `grid_portfolio.csv` | 설정별 tranche 포트폴리오 성과(롱숏 · 롱온리) |
| `grid_heatmap_*.png` | 예측기간 × 윈도우 히트맵(색 = Buy & Hold 대비, 숫자 = 지표 값, 굵은 테두리 = n ≈ s) |
| `wf_metrics*.csv`, `wf_portfolio*.csv` | walk-forward 논문 지표 · 포트폴리오(예측기간별) |
| `wf_horizon_blend.csv`, `wf_equity_*.png` | 예측기간 1–30 전략을 동일가중한 포트폴리오와 누적수익률 |
| `*_predictions.csv.gz` | 일별 예측 · 실제 레이블 · s일 수익률 |

## 구조

```
run_ti_forecast.py      진입점 (CLI)
ti_forecast/
  data.py               DataGuide 엑셀 / csv → 시가·고가·저가·종가
  indicators.py         10개 기술지표 (OHLC 원식 · 종가 전용)
  models.py             SVM · ANN · kNN, purged k-fold 하이퍼파라미터 선택
  experiment.py         레이블 · 분할 · 논문 지표 · tranche 백테스트 · 그리드 · walk-forward
  plotting.py           히트맵 · 누적수익률
tests/                  지표 식 · 미래 정보 미사용 · 분할 · 백테스트 · 전체 실행
```

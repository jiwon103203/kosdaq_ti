"""기술지표 + 머신러닝 방향 예측 (Shynkevich et al., "Forecasting price movements using technical
indicators: Investigating the impact of varying input window length", Neurocomputing 2017) 재현.

진입점은 저장소 루트의 ``run_ti_forecast.py``.

모듈 구성
    data         DataGuide 엑셀 / csv → 시가·고가·저가·종가
    indicators   논문의 10개 기술지표 (OHLC 원식 · 종가 전용 = 고가 = 저가 = 종가)
    models       SVM(sigmoid) · ANN · kNN, 시계열 purged k-fold 로 하이퍼파라미터 선택
    experiment   레이블 · 학습/평가 분할 · 논문 지표 · 포트폴리오 백테스트 · 그리드 · walk-forward
"""

"""논문(Sec. 5.3)의 10개 기술지표. 모든 지표가 같은 입력 윈도우 길이 ``n`` 을 쓴다.

``close_only=True`` 면 고가 = 저가 = 종가로 두고 같은 식을 계산한다. 그러면 true range 는 |ΔC|, 방향성 움직임은
max(±ΔC, 0), 고가·저가 채널은 종가 채널이 되어 종가만 있는 구간(코스닥150 2010-01 ~ 2015-07)에도 지표를 만들 수 있다.

    지표       OHLC 원식                                   종가 전용 (H = L = C)
    sma        C / SMA_n(C) - 1                            같음
    ema        C / EMA_n(C) - 1                            같음
    atr        Wilder_n(TR) / C                            Wilder_n(|ΔC|) / C
    adx        Wilder_n(DX), DX = 100|DI+ - DI-|/(DI+ + DI-) DM = max(±ΔC, 0)
    cci        (TP - SMA_n TP) / (0.015 · 평균편차), TP = (H+L+C)/3   TP = C
    roc        C_t / C_{t-n} - 1                           같음
    rsi        100 · Wilder_n(상승) / (Wilder_n(상승) + Wilder_n(하락))  같음
    willr      -100 (HH_n - C) / (HH_n - LL_n)              HH, LL = n일 최고·최저 종가
    stoch_k    100 (C - LL_n) / (HH_n - LL_n)               〃
    stoch_d    EMA_3(stoch_k)                              〃

논문은 SMA·EMA·ATR 을 가격 수준 그대로 쓰고 학습구간 평균·분산으로 표준화하지만, 한 지수를 16년 동안 쓰면 평가구간 가격이
학습구간 범위를 벗어나므로 종가 대비 비율로 바꿨다(정보는 같고 정상성만 다르다). 평활은 논문이 계산에 쓴 TA-Lib 과 같이
ATR·RSI·DI·ADX 는 Wilder(alpha = 1/n), EMA 는 alpha = 2/(n+1) 이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

NAMES = ("sma", "ema", "atr", "adx", "cci", "roc", "rsi", "willr", "stoch_k", "stoch_d")


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _mean_abs_dev(x: pd.Series, n: int) -> pd.Series:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = sliding_window_view(x.to_numpy(dtype=float), n)
        out[n - 1:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(out, index=x.index)


def compute_indicators(df: pd.DataFrame, n: int, close_only: bool = False) -> pd.DataFrame:
    """``df`` 의 ``close`` (+ ``high``, ``low``) 로 10개 지표를 계산한다. 앞쪽 계산 불가 구간은 NaN."""
    if n < 2:
        raise ValueError(f"input window must be >= 2, got {n}")
    c = df["close"].astype(float)
    h = c if close_only else df["high"].astype(float)
    l = c if close_only else df["low"].astype(float)
    prev_c = c.shift(1)

    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1, skipna=False)
    atr = _wilder(tr, n)

    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).mask(up.isna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).mask(down.isna())
    plus_di = 100.0 * _wilder(plus_dm, n) / atr
    minus_di = 100.0 * _wilder(minus_dm, n) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _wilder(dx, n)

    tp = (h + l + c) / 3.0
    cci = (tp - tp.rolling(n).mean()) / (0.015 * _mean_abs_dev(tp, n))

    delta = c.diff()
    gain, loss = _wilder(delta.clip(lower=0.0), n), _wilder((-delta).clip(lower=0.0), n)
    rsi = 100.0 * gain / (gain + loss)

    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    stoch_k = 100.0 * (c - ll) / (hh - ll)

    out = pd.DataFrame({
        "sma": c / c.rolling(n).mean() - 1.0,
        "ema": c / _ema(c, n) - 1.0,
        "atr": atr / c,
        "adx": adx,
        "cci": cci,
        "roc": c / c.shift(n) - 1.0,
        "rsi": rsi,
        "willr": -100.0 * (hh - c) / (hh - ll),
        "stoch_k": stoch_k,
        "stoch_d": _ema(stoch_k, 3),
    }, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)

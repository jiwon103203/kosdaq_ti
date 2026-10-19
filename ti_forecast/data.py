"""시가·고가·저가·종가 로드.

DataGuide 엑셀(위쪽 메타데이터 몇 줄 뒤 ``D A T E | 저가지수 | 고가지수 | 시가지수 | 종가지수`` 머리행)과
``date, open, high, low, close`` 형태의 일반 csv/xlsx 를 모두 읽는다. 시가·고가·저가는 없어도 되고(종가 전용 지표만
쓸 수 있음), 종가가 있는 행만 남긴다.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

ENCODINGS = ("utf-8-sig", "cp949", "euc-kr")  # utf-8-sig 는 BOM 없는 utf-8 도 읽는다

# 정규화된(소문자, 공백/기호 제거) 열 이름 후보. 앞에 있을수록 우선.
DATE_ALIASES = ("date", "dates", "datetime", "timestamp", "time", "날짜", "일자", "일시", "기준일", "기준일자",
                "거래일", "거래일자", "basdt", "trddd")
CLOSE_ALIASES = ("close", "adjclose", "adjustedclose", "closeprice", "price", "last", "pxlast", "종가", "수정종가",
                 "종가지수", "현재가", "지수", "가격", "clpr")
OPEN_ALIASES = ("open", "openprice", "시가", "시가지수", "수정시가")
HIGH_ALIASES = ("high", "highprice", "고가", "고가지수", "수정고가")
LOW_ALIASES = ("low", "lowprice", "저가", "저가지수", "수정저가")

_NA_TOKENS = {"", "-", "--", "—", ".", "nan", "NaN", "NAN", "N/A", "n/a", "#N/A", "NA", "null", "None"}


# ---------------------------------------------------------------------------
# 파일 · 열 읽기
# ---------------------------------------------------------------------------

def _sniff_delimiter(first_line: str) -> str:
    counts = {sep: first_line.count(sep) for sep in (",", "\t", ";", "|")}
    sep, n = max(counts.items(), key=lambda kv: kv[1])
    return sep if n > 0 else ","


def read_text_table(path: str) -> pd.DataFrame:
    """csv/tsv/txt 를 문자열로 읽는다. utf-8 → cp949 → euc-kr 순으로 인코딩을 시도한다."""
    ext = os.path.splitext(path)[1].lower()
    last_err: Optional[Exception] = None
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                first_line = f.readline()
            sep = "\t" if ext == ".tsv" else _sniff_delimiter(first_line)
            return pd.read_csv(path, encoding=enc, sep=sep, dtype=str, keep_default_na=False)
        except UnicodeDecodeError as e:
            last_err = e
    raise ValueError(f"{path}: 인코딩 {ENCODINGS} 중 어느 것으로도 읽을 수 없습니다") from last_err


def to_numeric(s: pd.Series) -> pd.Series:
    """'1,234.5', '3.5%', '(1.2)', '-' 같은 문자열을 float 로 변환한다 (% 는 기호만 제거)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype("string").str.strip()
    t = t.str.replace(r"[,%\s₩$]", "", regex=True)
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # 회계식 음수
    t = t.mask(t.isin(_NA_TOKENS))
    return pd.to_numeric(t, errors="coerce").astype(float)


def parse_dates(s: pd.Series) -> pd.DatetimeIndex:
    """'2024-01-02', '2024/01/02', '2024.01.02', '20240102', '2024년 1월 2일', 엑셀 serial 을 모두 처리한다."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.DatetimeIndex(s).normalize()
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
        valid = v.dropna()
        if len(valid) and valid.between(19000101, 21001231).all():
            return pd.DatetimeIndex(pd.to_datetime(v.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce"))
        if len(valid) and valid.between(1, 80000).all():
            return pd.DatetimeIndex(pd.to_datetime(v, unit="D", origin="1899-12-30", errors="coerce")).normalize()
    t = s.astype("string").str.strip()
    t = t.str.replace(r"\s*([./년월일])\s*", r"\1", regex=True)  # '2024. 1. 2.' → '2024.1.2.'
    ymd = t.str.fullmatch(r"\d{8}").fillna(False)
    t = t.mask(ymd, t.str.slice(0, 4) + "-" + t.str.slice(4, 6) + "-" + t.str.slice(6, 8))
    t = t.str.replace(r"[./년월]", "-", regex=True).str.replace("일", "", regex=False).str.strip("-")
    return pd.DatetimeIndex(pd.to_datetime(t, errors="coerce", format="mixed")).normalize()


def _norm(name) -> str:
    return re.sub(r"[\s_\-\.\(\)\[\]%/]", "", str(name)).lower()


def find_column(columns: Iterable, aliases: Sequence[str], exclude: Iterable = ()) -> Optional[str]:
    """열 이름을 자동 인식한다: 정확히 일치 → 부분 일치(3글자 이상 또는 한글 후보만) 순."""
    exclude = {c for c in exclude if c is not None}
    cols = [c for c in columns if c not in exclude]
    normed = {c: _norm(c) for c in cols}
    for alias in aliases:
        for c in cols:
            if normed[c] == alias:
                return c
    for alias in aliases:
        if len(alias) < 3 and alias.isascii():
            continue
        for c in cols:
            if alias in normed[c]:
                return c
    return None


def _promote_header(raw: pd.DataFrame) -> pd.DataFrame:
    """머리행이 데이터 중간에 있으면(DataGuide) 첫 열이 날짜 이름인 행을 머리행으로 올린다."""
    if find_column(raw.columns, DATE_ALIASES) is not None:
        return raw
    first = raw.iloc[:, 0].map(_norm)
    hits = first.index[first.isin(DATE_ALIASES)]
    if len(hits) == 0:
        return raw
    row = raw.index.get_loc(hits[0])
    out = raw.iloc[row + 1:].copy()
    out.columns = [str(c).strip() for c in raw.iloc[row]]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 가격
# ---------------------------------------------------------------------------

def load_ohlc(path: str, sheet=None) -> pd.DataFrame:
    """``open, high, low, close`` 열(없는 가격은 NaN)과 DatetimeIndex 를 가진 DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.splitext(path)[1].lower() in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, header=None)
        raw.columns = [f"col{i}" for i in range(raw.shape[1])]
    else:
        raw = read_text_table(path)
    raw = _promote_header(raw)
    raw.columns = [str(c).strip() for c in raw.columns]

    dcol = find_column(raw.columns, DATE_ALIASES) or raw.columns[0]
    cols = {"open": find_column(raw.columns, OPEN_ALIASES, (dcol,)),
            "high": find_column(raw.columns, HIGH_ALIASES, (dcol,)),
            "low": find_column(raw.columns, LOW_ALIASES, (dcol,))}
    cols["close"] = find_column(raw.columns, CLOSE_ALIASES, (dcol, *cols.values()))
    if cols["close"] is None:
        raise ValueError(f"{path}: 종가 열을 찾지 못했습니다. 열: {list(raw.columns)}")

    df = pd.DataFrame({k: (to_numeric(raw[c]).to_numpy() if c is not None else float("nan"))
                       for k, c in cols.items()}, index=parse_dates(raw[dcol]))
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"].notna() & (df["close"] > 0)]
    df.index.name = "date"
    ohlc = df[["open", "high", "low"]].notna().all(axis=1)
    logger.info("loaded %s: %d rows close %s ~ %s, OHLC from %s (columns %s)", path, len(df),
                df.index.min().date(), df.index.max().date(),
                df.index[ohlc.to_numpy()].min().date() if ohlc.any() else None, cols)
    return df


def ohlc_start(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """시가·고가·저가·종가가 모두 있는 첫 날짜 (이후 구간은 끊김 없이 모두 있어야 한다)."""
    full = df[["open", "high", "low", "close"]].notna().all(axis=1)
    if not full.any():
        return None
    start = full.index[full.to_numpy()].min()
    if not full.loc[start:].all():
        missing = full.loc[start:][~full.loc[start:]].index
        raise ValueError(f"OHLC 가 {start.date()} 이후 {len(missing)}일 비어 있습니다 (첫 날짜 {missing[0].date()})")
    return start

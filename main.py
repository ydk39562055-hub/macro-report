"""
매크로 리포트 생성 시스템
헤지펀드 수준의 팩트 기반 매크로 분석 + AI 전문가 토론
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import google.generativeai as genai

# ── 설정 ──────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
FRED_API_KEY   = os.environ.get("FRED_API_KEY")

TICKERS = {
    "nasdaq":        "^IXIC",
    "bitcoin":       "BTC-USD",
    "semiconductor": "SOXX",
    "vix":           "^VIX",
    "treasury_10y":  "^TNX",
    "dollar_index":  "DX-Y.NYB",
}

FRED_SERIES = {
    "hy_spread":          "BAMLH0A0HYM2",   # 하이일드 스프레드
    "fed_funds_rate":     "FEDFUNDS",        # 기준금리
    "cpi_yoy":            "CPIAUCSL",        # CPI
    "unemployment":       "UNRATE",          # 실업률
    "m2_money_supply":    "M2SL",            # M2 통화량
    "consumer_sentiment": "UMCSENT",         # 소비자 심리
}


# ── 상관계수 엔진 ──────────────────────────────────────────────────────────────
class CorrelationEngine:
    """60일 이동 상관계수 계산 전용 클래스"""

    def __init__(self, window: int = 60):
        self.window = window

    def _returns(self, series: pd.Series) -> pd.Series:
        return series.pct_change().dropna()

    def rolling_corr(self, s1: pd.Series, s2: pd.Series) -> float | None:
        try:
            df = pd.DataFrame({"a": s1, "b": s2}).dropna()
            if len(df) < self.window:
                return None
            val = df["a"].rolling(self.window).corr(df["b"]).iloc[-1]
            return round(float(val), 4) if not np.isnan(val) else None
        except Exception:
            return None

    def compute_all(self, price_data: dict) -> dict:
        returns = {
            name: self._returns(series)
            for name, series in price_data.items()
            if series is not None and len(series) > 1
        }
        keys = list(returns.keys())
        result = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                c = self.rolling_corr(returns[k1], returns[k2])
                if c is not None:
                    result[f"{k1}_vs_{k2}"] = c
        return result


# ── 데이터 수집 ────────────────────────────────────────────────────────────────
def fetch_market_data(tickers: dict, period: str = "90d") -> dict:
    data = {}
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if df.empty:
                print(f"[WARN] {sym}: 빈 데이터")
                data[name] = None
            else:
                data[name] = df["Close"].squeeze()
                print(f"[OK]   {sym}: {len(df)}일")
        except Exception as e:
            print(f"[ERR]  {sym}: {e}")
            data[name] = None
    return data


def fetch_fred_data(series_map: dict, api_key: str) -> dict:
    if not api_key:
        print("[WARN] FRED_API_KEY 없음 → FRED 스킵")
        return {k: "Data Unavailable" for k in series_map}

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except ImportError:
        print("[ERR]  fredapi 패키지 없음")
        return {k: "Data Unavailable" for k in series_map}
    except Exception as e:
        print(f"[ERR]  FRED 연결 실패: {e}")
        return {k: "Data Unavailable" for k in series_map}

    result = {}
    for name, sid in series_map.items():
        try:
            s = fred.get_series(sid, observation_start="2024-01-01").dropna()
            latest, prev = float(s.iloc[-1]), float(s.iloc[-2]) if len(s) > 1 else float(s.iloc[-1])
            result[name] = {
                "series_id": sid,
                "value":     round(latest, 4),
                "prev":      round(prev, 4),
                "change":    round(latest - prev, 4),
            }
            print(f"[OK]   FRED {sid}: {latest:.4f}")
        except Exception as e:
            print(f"[ERR]  FRED {sid}: {e}")
            result[name] = "Data Unavailable"
    return result


# ── 인과관계 트리거 분석 ───────────────────────────────────────────────────────
def analyze_triggers(market: dict, fred: dict) -> dict:
    triggers = {}

    # 1) 금리 변동폭 대비 나스닥 변동성 비율
    try:
        nq  = market.get("nasdaq")
        tsy = market.get("treasury_10y")
        if nq is not None and tsy is not None:
            nq_vol      = nq.pct_change().dropna().tail(20).std() * np.sqrt(252) * 100
            tsy_delta   = abs(tsy.diff().dropna().tail(5).mean())
            ratio       = float(nq_vol / tsy_delta) if tsy_delta else 0
            triggers["rate_nasdaq_vol_ratio"] = round(ratio, 2)
        else:
            triggers["rate_nasdaq_vol_ratio"] = "Data Unavailable"
    except Exception:
        triggers["rate_nasdaq_vol_ratio"] = "Data Unavailable"

    # 2) VIX 패닉 셀링 체크 (1.5σ 초과)
    try:
        nq  = market.get("nasdaq")
        vix = market.get("vix")
        if nq is not None and vix is not None:
            nq_ret  = nq.pct_change().dropna().tail(60)
            vix_ret = vix.pct_change().dropna().tail(60)
            down_idx        = nq_ret[nq_ret < 0].index
            vix_on_down     = vix_ret[vix_ret.index.isin(down_idx)]
            if len(vix_on_down) > 5:
                mu, sigma       = vix_on_down.mean(), vix_on_down.std()
                latest_change   = float(vix_ret.iloc[-1])
                z               = (latest_change - mu) / sigma if sigma > 0 else 0
                triggers["panic_selling"] = {
                    "vix_z_score":           round(float(z), 2),
                    "is_panic":              bool(z > 1.5),
                    "latest_vix_change_pct": round(latest_change * 100, 2),
                }
            else:
                triggers["panic_selling"] = "Insufficient Data"
        else:
            triggers["panic_selling"] = "Data Unavailable"
    except Exception:
        triggers["panic_selling"] = "Data Unavailable"

    # 3) 하이일드 스프레드 자금난 경보
    try:
        hy = fred.get("hy_spread")
        if isinstance(hy, dict):
            v = hy["value"]
            triggers["liquidity_crisis"] = {
                "hy_spread": v,
                "change":    hy["change"],
                "alert":     bool(v > 3.5),
                "severity":  "HIGH" if v > 5.0 else ("MEDIUM" if v > 3.5 else "LOW"),
            }
        else:
            triggers["liquidity_crisis"] = "Data Unavailable"
    except Exception:
        triggers["liquidity_crisis"] = "Data Unavailable"

    return triggers


# ── 최신 값 추출 ───────────────────────────────────────────────────────────────
def latest_values(market: dict) -> dict:
    out = {}
    for name, series in market.items():
        if series is not None and len(series) > 0:
            v    = float(series.iloc[-1])
            prev = float(series.iloc[-2]) if len(series) > 1 else v
            chg  = (v - prev) / prev * 100 if prev else 0
            out[name] = {"value": round(v, 4), "change_pct": round(chg, 2)}
        else:
            out[name] = "Data Unavailable"
    return out


# ── AI 전문가 토론 생성 ────────────────────────────────────────────────────────
def generate_debate(facts: dict, api_key: str) -> dict:
    if not api_key:
        return {k: "GOOGLE_API_KEY 없음" for k in ("analyst_a", "analyst_b", "analyst_c")}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
당신은 글로벌 헤지펀드 투자 전략 회의를 주재하는 AI입니다.
아래 팩트 데이터를 근거로 3명의 애널리스트가 의견을 제시합니다.

★ 절대 원칙
- 추측성 표현("~할 수 있다", "~일지도") 사용 금지
- 수치(%) 반드시 인용
- 전문 용어 사용 (리스크 오프, 크레딧 스프레드, σ 등)
- 각 발언 200자 이내

팩트 데이터:
{json.dumps(facts, ensure_ascii=False, indent=2)}

[애널리스트 A – 리스크 매니저]
VIX·하이일드 스프레드 기반으로 리스크 오프 강도를 평가하고 현금 확보·헤지 논거 제시.

[애널리스트 B – 성장 전략가]
SOXX·나스닥 데이터 기반으로 AI·기술 섹터 구조적 성장 팩트와 매수 논거 제시.

[애널리스트 C – 퀀트 전략가]
상관계수(특히 0.9 초과 여부)를 분석하고 통계적 안착 전까지의 포지션 전략 제시.

반드시 아래 JSON 형식으로만 응답:
{{
  "analyst_a": "...",
  "analyst_b": "...",
  "analyst_c": "..."
}}
"""
        resp = model.generate_content(prompt)
        text = resp.text.strip()

        # 마크다운 코드블록 제거
        for fence in ("```json", "```"):
            if fence in text:
                text = text.split(fence)[1].split("```")[0].strip()
                break

        return json.loads(text)

    except Exception as e:
        print(f"[ERR]  AI 토론 생성 실패: {e}")
        return {k: f"생성 실패: {e}" for k in ("analyst_a", "analyst_b", "analyst_c")}


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(f"  매크로 리포트 | {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 55)

    # 1. 수집
    print("\n[1] 시장 데이터 수집...")
    market = fetch_market_data(TICKERS)

    print("\n[2] FRED 매크로 데이터 수집...")
    fred = fetch_fred_data(FRED_SERIES, FRED_API_KEY)

    # 2. 분석
    print("\n[3] 상관계수 계산...")
    corr = CorrelationEngine(60).compute_all(market)

    print("\n[4] 트리거 분석...")
    triggers = analyze_triggers(market, fred)

    lv = latest_values(market)

    # 3. AI 토론
    facts = {
        "date":            datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "market":          lv,
        "fred":            fred,
        "correlations_60d": corr,
        "triggers":        triggers,
    }
    print("\n[5] AI 전문가 토론 생성...")
    debate = generate_debate(facts, GOOGLE_API_KEY)

    # 4. 매크로 요약 한 줄
    def _v(d, k, sub="value"):
        x = d.get(k, {})
        return x.get(sub, "N/A") if isinstance(x, dict) else "N/A"

    macro_summary = (
        f"VIX {_v(lv,'vix')} ({_v(lv,'vix','change_pct')}%) | "
        f"HY Spread {_v(fred,'hy_spread')} | "
        f"Nasdaq {_v(lv,'nasdaq','change_pct')}% | "
        f"BTC {_v(lv,'bitcoin','change_pct')}%"
    )

    # 5. 저장
    kst = datetime.timezone(datetime.timedelta(hours=9))
    report = {
        "date":              datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST"),
        "macro_summary":     macro_summary,
        "expert_debate":     debate,
        "asset_correlations": corr,
        "key_metrics": {
            "vix":           lv.get("vix", "Data Unavailable"),
            "hy_spread":     fred.get("hy_spread", "Data Unavailable"),
            "nasdaq":        lv.get("nasdaq", "Data Unavailable"),
            "bitcoin":       lv.get("bitcoin", "Data Unavailable"),
            "semiconductor": lv.get("semiconductor", "Data Unavailable"),
            "treasury_10y":  lv.get("treasury_10y", "Data Unavailable"),
            "dollar_index":  lv.get("dollar_index", "Data Unavailable"),
        },
        "fred_macro":        fred,
        "causal_triggers":   triggers,
    }

    os.makedirs("reports", exist_ok=True)
    path = "reports/latest_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] {path} 저장됨")
    print(f"       {macro_summary}")
    print("=" * 55)


if __name__ == "__main__":
    main()

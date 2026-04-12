"""
매크로 리포트 생성 시스템
헤지펀드 수준의 팩트 기반 매크로 분석 + AI 전문가 토론
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from groq import Groq

# ── 설정 ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
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
    "core_cpi":           "CPILFESL",        # 근원 CPI (식품·에너지 제외)
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
def fetch_market_data(tickers: dict, period: str = "1y") -> dict:
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


# ── Fear & Greed Index 수집 ───────────────────────────────────────────────────
def fetch_fear_greed() -> dict:
    result = {}

    # 1) CNN Fear & Greed Index (주식시장)
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        score = data["fear_and_greed"]["score"]
        rating = data["fear_and_greed"]["rating"]
        prev_close = data["fear_and_greed"]["previous_close"]
        result["cnn_fear_greed"] = {
            "score": round(float(score), 1),
            "rating": rating,
            "previous_close": round(float(prev_close), 1),
            "change": round(float(score) - float(prev_close), 1),
        }
        print(f"[OK]   CNN Fear & Greed: {score} ({rating})")
    except Exception as e:
        print(f"[ERR]  CNN Fear & Greed: {e}")
        result["cnn_fear_greed"] = "Data Unavailable"

    # 2) Crypto Fear & Greed Index (암호화폐)
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=2",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        today = data[0]
        yesterday = data[1] if len(data) > 1 else data[0]
        score = int(today["value"])
        prev = int(yesterday["value"])
        result["crypto_fear_greed"] = {
            "score": score,
            "rating": today["value_classification"],
            "previous": prev,
            "change": score - prev,
        }
        print(f"[OK]   Crypto Fear & Greed: {score} ({today['value_classification']})")
    except Exception as e:
        print(f"[ERR]  Crypto Fear & Greed: {e}")
        result["crypto_fear_greed"] = "Data Unavailable"

    return result


# ── 강세/약세장 조건 체크 (마크 미네르비니 기반) ─────────────────────────────────
def check_bull_bear_conditions(market: dict, fred: dict, fear_greed: dict) -> dict:
    bull = {}
    bear = {}

    # ── 강세 조건 (7개) ──

    # 1. VIX < 20 (공포 지수 안정)
    try:
        vix = market.get("vix")
        if vix is not None:
            v = float(vix.iloc[-1])
            bull["vix_stable"] = {
                "name": "VIX 안정권",
                "desc": f"VIX {v:.1f} (기준 < 20)",
                "met": v < 20,
            }
    except Exception:
        pass

    # 2. 나스닥 200MA 상회
    try:
        nq = market.get("nasdaq")
        if nq is not None and len(nq) >= 200:
            ma200 = float(nq.rolling(200).mean().iloc[-1])
            cur = float(nq.iloc[-1])
            bull["nasdaq_200ma"] = {
                "name": "나스닥 200MA 상회",
                "desc": f"현재 {cur:,.0f} vs 200MA {ma200:,.0f} ({(cur/ma200-1)*100:+.1f}%)",
                "met": cur > ma200,
            }
    except Exception:
        pass

    # 3. 신용시장 안정 (HY Spread < 3.5)
    try:
        hy = fred.get("hy_spread")
        if isinstance(hy, dict):
            v = hy["value"]
            bull["credit_stable"] = {
                "name": "신용시장 안정",
                "desc": f"HY 스프레드 {v} (기준 < 3.5)",
                "met": v < 3.5,
            }
    except Exception:
        pass

    # 4. 고용 견조 (실업률 < 5%)
    try:
        unemp = fred.get("unemployment")
        if isinstance(unemp, dict):
            v = unemp["value"]
            bull["employment"] = {
                "name": "고용 견조",
                "desc": f"실업률 {v}% (기준 < 5%)",
                "met": v < 5.0,
            }
    except Exception:
        pass

    # 5. 소비심리 안정 (> 50)
    try:
        cs = fred.get("consumer_sentiment")
        if isinstance(cs, dict):
            v = cs["value"]
            bull["consumer"] = {
                "name": "소비심리 안정",
                "desc": f"소비자심리지수 {v} (기준 > 50)",
                "met": v > 50,
            }
    except Exception:
        pass

    # 6. 반도체 200MA 상회
    try:
        soxx = market.get("semiconductor")
        if soxx is not None and len(soxx) >= 200:
            ma200 = float(soxx.rolling(200).mean().iloc[-1])
            cur = float(soxx.iloc[-1])
            bull["semi_200ma"] = {
                "name": "반도체 200MA 상회",
                "desc": f"SOXX {cur:,.1f} vs 200MA {ma200:,.1f} ({(cur/ma200-1)*100:+.1f}%)",
                "met": cur > ma200,
            }
    except Exception:
        pass

    # 7. 인플레 둔화 (근원 CPI MoM < 0.3%)
    try:
        core = fred.get("core_cpi")
        if isinstance(core, dict) and core["prev"]:
            mom = core["change"] / core["prev"] * 100
            bull["inflation_cool"] = {
                "name": "인플레 둔화",
                "desc": f"근원CPI MoM {mom:.2f}% (기준 < 0.3%)",
                "met": mom < 0.3,
            }
    except Exception:
        pass

    # ── 약세 조건 (4개) ──

    # 1. VIX > 25
    try:
        vix = market.get("vix")
        if vix is not None:
            v = float(vix.iloc[-1])
            bear["vix_fear"] = {
                "name": "VIX 공포 확대",
                "desc": f"VIX {v:.1f} (기준 > 25)",
                "met": v > 25,
            }
    except Exception:
        pass

    # 2. 신용 경색 (HY Spread > 4.0)
    try:
        hy = fred.get("hy_spread")
        if isinstance(hy, dict):
            v = hy["value"]
            bear["credit_stress"] = {
                "name": "신용 경색",
                "desc": f"HY 스프레드 {v} (기준 > 4.0)",
                "met": v > 4.0,
            }
    except Exception:
        pass

    # 3. 나스닥 200MA 하회
    try:
        nq = market.get("nasdaq")
        if nq is not None and len(nq) >= 200:
            ma200 = float(nq.rolling(200).mean().iloc[-1])
            cur = float(nq.iloc[-1])
            bear["nasdaq_below"] = {
                "name": "나스닥 200MA 하회",
                "desc": f"현재 {cur:,.0f} vs 200MA {ma200:,.0f}",
                "met": cur < ma200,
            }
    except Exception:
        pass

    # 4. BTC 200MA 하회
    try:
        btc = market.get("bitcoin")
        if btc is not None and len(btc) >= 200:
            ma200 = float(btc.rolling(200).mean().iloc[-1])
            cur = float(btc.iloc[-1])
            bear["btc_below"] = {
                "name": "BTC 200MA 하회",
                "desc": f"BTC {cur:,.0f} vs 200MA {ma200:,.0f}",
                "met": cur < ma200,
            }
    except Exception:
        pass

    # 집계
    bull_met = sum(1 for c in bull.values() if c["met"])
    bull_total = len(bull)
    bear_met = sum(1 for c in bear.values() if c["met"])
    bear_total = len(bear)

    confidence = round(bull_met / bull_total * 100) if bull_total > 0 else 50

    if bull_met >= 5 and bear_met <= 1:
        regime, regime_kr = "RISK_ON", "강세장"
    elif bear_met >= 3:
        regime, regime_kr = "RISK_OFF", "약세장"
    else:
        regime, regime_kr = "NEUTRAL", "중립"

    return {
        "regime": regime,
        "regime_kr": regime_kr,
        "confidence": confidence,
        "bull": {"met": bull_met, "total": bull_total, "details": bull},
        "bear": {"met": bear_met, "total": bear_total, "details": bear},
    }


def generate_briefing(facts: dict, regime: dict, api_key: str) -> dict:
    """시황 브리핑 생성 — 차근이 스타일"""
    if not api_key:
        return {"holder_advice": [], "cash_advice": [], "watch_points": []}

    try:
        client = Groq(api_key=api_key)

        prompt = f"""아래 데이터를 전부 읽고 시황 브리핑을 JSON으로 작성해.

[데이터]
시장 판정: {regime['regime_kr']} (강세 {regime['bull']['met']}/{regime['bull']['total']}, 약세 {regime['bear']['met']}/{regime['bear']['total']})
{json.dumps(facts, ensure_ascii=False, indent=2)}
{json.dumps(regime, ensure_ascii=False, indent=2)}

[나쁜 예시 — 이렇게 쓰면 안 됨]
- "시장의 변동성을 주시하면서 적절한 타이밍을 기다리세요" → 뻔한 말
- "신중하게 결정하세요" → 아무 정보 없음
- "긍정적인 전망을 가지고 투자에 대한 준비를 하세요" → 빈말
- "안정적인 상황이니 투자할 때 주의하세요" → 모순

[좋은 예시 — 이런 식으로 써야 함]
- "강세 지표가 대부분 통과했지만, 공포탐욕지수가 16으로 극단적 공포예요. 펀더멘털과 심리가 엇갈리는 구간이라 한쪽만 믿기엔 위험해요."
- "BTC가 200일 평균(87,873)보다 19% 아래인 70,987이에요. 200일선 회복 전까지는 코인 쪽 추가 매수는 쉬어가는 게 나아요."
- "근원CPI가 전월대비 0.20%인데, 이걸 연환산하면 2.4%예요. 연준 목표 2%보다 높아서 금리 인하 기대는 아직 이른 편이에요."
- "VIX가 19.2로 20 바로 밑이에요. 20 넘어가면 분위기가 확 바뀔 수 있으니, 22까지 오르면 주식 비중을 5~10% 줄여보세요."

[규칙]
1. 한자(株, 債) 절대 금지
2. 영어 금지
3. "~하세요", "~합니다" 금지 → "~해요", "~이에요", "~좋아요", "~나아요" 등 친근체
4. 모든 문장에 수치 필수
5. 데이터끼리 모순되면 솔직히 말할 것
6. 각 항목 1~2문장

JSON만 응답:
{{
  "holder_advice": ["조언1", "조언2", "조언3"],
  "cash_advice": ["조언1", "조언2", "조언3"],
  "watch_points": ["체크1", "체크2", "체크3"]
}}"""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        return json.loads(resp.choices[0].message.content.strip())

    except Exception as e:
        print(f"[ERR]  시황 브리핑 생성 실패: {e}")
        return {"holder_advice": [], "cash_advice": [], "watch_points": []}


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
        print("[ERR]  GROQ_API_KEY 환경변수 없음")
        return {k: "GROQ_API_KEY 없음" for k in ("analyst_a", "analyst_b", "analyst_c")}

    print(f"[OK]   GROQ_API_KEY 확인됨 (길이: {len(api_key)})")

    try:
        client = Groq(api_key=api_key)

        prompt = f"""당신은 글로벌 헤지펀드 투자 전략 회의를 주재하는 AI입니다.
아래 팩트 데이터를 근거로 3명의 애널리스트가 의견을 제시합니다.

★ 작성 원칙
- 투자 초보자도 이해할 수 있도록 쉬운 말로 설명 (전문 용어 사용 시 괄호로 쉬운 설명 추가)
- 수치를 반드시 인용하고 그 의미를 해석
- 각 애널리스트 발언은 300~400자 분량
- 발언 마지막에 반드시 【결론: 매수】 또는 【결론: 매도】 또는 【결론: 관망】으로 명확히 끝낼 것
- 추측성 표현("~할 수 있다") 금지, 데이터 기반 단정적 표현 사용

팩트 데이터:
{json.dumps(facts, ensure_ascii=False, indent=2)}

[애널리스트 A – 리스크 매니저]
VIX(공포 지수)와 하이일드 스프레드(고위험 채권 금리차)를 분석해서, 지금 시장이 얼마나 위험한 상태인지 설명하고 현금 비중을 높여야 하는지 판단.

[애널리스트 B – 성장 전략가]
나스닥·반도체(SOXX) 데이터를 보고 AI·기술 섹터에 지금 투자해도 되는지, 매수 타이밍인지 판단.

[애널리스트 C – 퀀트 전략가]
자산 간 상관계수(함께 움직이는 정도)를 분석해서 분산투자가 잘 되고 있는지, 지금 포지션(투자 비중)을 유지할지 조정할지 판단.

반드시 아래 JSON 형식으로만 응답:
{{
  "analyst_a": "...",
  "analyst_b": "...",
  "analyst_c": "..."
}}"""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        text = resp.choices[0].message.content.strip()
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

    print("\n[2.5] Fear & Greed Index 수집...")
    fear_greed = fetch_fear_greed()

    # 2. 분석
    print("\n[3] 상관계수 계산...")
    corr = CorrelationEngine(60).compute_all(market)

    print("\n[4] 트리거 분석...")
    triggers = analyze_triggers(market, fred)

    print("\n[4.5] 강세/약세장 조건 체크...")
    regime = check_bull_bear_conditions(market, fred, fear_greed)
    print(f"       → {regime['regime_kr']} | 확신도 {regime['confidence']}% | "
          f"강세 {regime['bull']['met']}/{regime['bull']['total']} "
          f"약세 {regime['bear']['met']}/{regime['bear']['total']}")

    lv = latest_values(market)

    # 3. AI 토론
    facts = {
        "date":            datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "market":          lv,
        "fred":            fred,
        "correlations_60d": corr,
        "triggers":        triggers,
        "fear_greed":      fear_greed,
        "regime":          regime,
    }
    print("\n[5] AI 전문가 토론 생성...")
    debate = generate_debate(facts, GROQ_API_KEY)

    print("\n[6] 시황 브리핑 생성...")
    briefing = generate_briefing(facts, regime, GROQ_API_KEY)

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
        "market_briefing": {
            "regime":        regime["regime"],
            "regime_kr":     regime["regime_kr"],
            "confidence":    regime["confidence"],
            "bull_score":    f"{regime['bull']['met']}/{regime['bull']['total']}",
            "bear_score":    f"{regime['bear']['met']}/{regime['bear']['total']}",
            "bull_details":  regime["bull"]["details"],
            "bear_details":  regime["bear"]["details"],
            "holder_advice": briefing.get("holder_advice", []),
            "cash_advice":   briefing.get("cash_advice", []),
            "watch_points":  briefing.get("watch_points", []),
        },
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
        "fear_greed":        fear_greed,
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

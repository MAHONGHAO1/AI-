from __future__ import annotations

import copy
import importlib.util
from io import StringIO
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def bootstrap_streamlit() -> None:
    """允许用户用 python 直接运行脚本时自动切换到 streamlit run。"""
    if os.getenv("STREAMLIT_BOOTSTRAPPED") == "1":
        return

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            return
    except Exception:
        pass

    env = os.environ.copy()
    env["STREAMLIT_BOOTSTRAPPED"] = "1"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
        ],
        env=env,
    )
    raise SystemExit


bootstrap_streamlit()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import OpenAI


st.set_page_config(
    page_title="AI 智选基金助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


FUNDS = [
    {"code": "006887", "name": "诺德新生活混合A"},
    {"code": "012920", "name": "易方达全球成长精选混合(QDII)A"},
    {"code": "023895", "name": "天弘上证科创板综合指数增强A"},
    {"code": "008585", "name": "华夏人工智能ETF联接A"},
    {"code": "270042", "name": "广发纳斯达克100ETF联接(QDII)A"},
    {"code": "010990", "name": "南方有色金属ETF联接E"},
    {"code": "014881", "name": "天弘中证机器人ETF联接C"},
    {"code": "009478", "name": "中银上海金ETF联接C"},
    {"code": "002611", "name": "博时黄金ETF联接C"},
    {"code": "024642", "name": "华夏中证银行ETF联接D"},
]


TYPE_KEYWORDS = {
    "全部": [],
    "科技": ["科技", "人工智能", "机器人", "科创"],
    "黄金": ["黄金", "上海金"],
    "银行": ["银行"],
    "纳斯达克": ["纳斯达克"],
    "有色金属": ["有色", "金属"],
    "QDII": ["QDII", "全球"],
    "其他": [],
}

UI_NOTICES: set[str] = set()

AI_DISCLAIMER = "以上内容由人工智能生成，不构成投资建议，请独立决策。"
COMPLIANCE_DISCLOSURE = "本工具仅用于学习交流，不提供任何形式的投资建议或交易执行。基金投资有风险，请根据自身风险承受能力谨慎决策。"

SCORE_DIMENSIONS = ["收益能力", "风险控制", "风险调整收益", "稳定性", "相对排名"]
DEFAULT_SCORE_WEIGHTS = {
    "收益能力": 0.30,
    "风险控制": 0.25,
    "风险调整收益": 0.25,
    "稳定性": 0.10,
    "相对排名": 0.10,
}

BENCHMARKS = {
    "沪深300": {"secid": "1.000300", "name": "沪深300"},
    "科创50": {"secid": "1.000688", "name": "科创50"},
    "纳斯达克100": {"secid": "100.NDX100", "name": "纳斯达克100"},
    "中证银行": {"secid": "0.399986", "name": "中证银行"},
    "中证有色金属": {"secid": "1.000819", "name": "中证有色金属"},
}


def default_holdings() -> list[dict[str, Any]]:
    """首次运行使用的示例组合，用户可在界面增删改。"""
    sample_values = {
        "006887": {"amount": 912.07, "hold_ret": 58.48, "cum_profit": 516.79},
        "012920": {"amount": 776.69, "hold_ret": 61.82, "cum_profit": 285.87},
        "023895": {"amount": 324.36, "hold_ret": 29.82, "cum_profit": 110.63},
        "008585": {"amount": 220.95, "hold_ret": 28.72, "cum_profit": 74.64},
        "270042": {"amount": 284.25, "hold_ret": 16.03, "cum_profit": 40.40},
        "010990": {"amount": 544.87, "hold_ret": 6.39, "cum_profit": 34.31},
        "014881": {"amount": 11.56, "hold_ret": 15.60, "cum_profit": -15.30},
        "009478": {"amount": 47.43, "hold_ret": -5.41, "cum_profit": 6.56},
        "002611": {"amount": 130.59, "hold_ret": -7.21, "cum_profit": 44.31},
        "024642": {"amount": 125.07, "hold_ret": -3.82, "cum_profit": -4.93},
    }
    return [
        {
            "code": fund["code"],
            "name": fund["name"],
            "amount": sample_values[fund["code"]]["amount"],
            "hold_ret": sample_values[fund["code"]]["hold_ret"],
            "cum_profit": sample_values[fund["code"]]["cum_profit"],
        }
        for fund in FUNDS
    ]


def normalize_fund_code(value: Any) -> str:
    text = str(value or "").strip()
    digits = re.sub(r"\D", "", text)
    return digits.zfill(6) if digits else ""


def normalize_holdings(raw_holdings: Any) -> list[dict[str, Any]]:
    """补齐组合字段并转换类型，支持增删改基金。"""
    if isinstance(raw_holdings, dict):
        source_rows = []
        default_names = {fund["code"]: fund["name"] for fund in FUNDS}
        for code, item in raw_holdings.items():
            source_rows.append(
                {
                    "code": code,
                    "name": item.get("name", default_names.get(str(code).zfill(6), "")),
                    "amount": item.get("amount", 0.0),
                    "hold_ret": item.get("hold_ret", 0.0),
                    "cum_profit": item.get("cum_profit", 0.0),
                }
            )
    elif isinstance(raw_holdings, list):
        source_rows = raw_holdings
    else:
        source_rows = default_holdings()

    normalized = []
    seen_codes = set()
    for item in source_rows:
        code = normalize_fund_code(item.get("code") or item.get("基金代码"))
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        name = str(item.get("name") or item.get("基金名称") or code).strip() or code
        normalized.append(
            {
                "code": code,
                "name": name,
                "amount": safe_number_value(item.get("amount", item.get("持有金额(元)", 0.0))),
                "hold_ret": safe_number_value(item.get("hold_ret", item.get("持有收益率(%)", 0.0))),
                "cum_profit": safe_number_value(item.get("cum_profit", item.get("累计收益(元)", 0.0))),
            }
        )
    return normalized


def holdings_to_json(holdings: Any) -> str:
    """稳定序列化组合，用作 cache_data 参数。"""
    return json.dumps(normalize_holdings(holdings), ensure_ascii=False, sort_keys=True)


def holdings_to_map(holdings: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        item["code"]: {
            "amount": float(item.get("amount", 0.0)),
            "hold_ret": float(item.get("hold_ret", 0.0)),
            "cum_profit": float(item.get("cum_profit", 0.0)),
        }
        for item in normalize_holdings(holdings)
    }


def default_snapshot_date() -> str:
    """示例持仓来自 2026-05-28 左右的账户快照。"""
    return "2026-05-28"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.3rem; }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(30,41,59,.92));
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 10px;
            padding: 18px 18px 14px;
        }
        [data-testid="stMetricLabel"] { color: #cbd5e1; }
        [data-testid="stMetricValue"] { color: #f8fafc; font-size: 1.45rem; }
        .hint-card {
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 10px;
            padding: 14px 16px;
            background: rgba(15,23,42,.72);
        }
        .small-muted { color: #94a3b8; font-size: 13px; }
        .compliance-footer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 999;
            background: rgba(15, 23, 42, .96);
            border-top: 1px solid rgba(148,163,184,.30);
            color: #cbd5e1;
            padding: 8px 18px;
            font-size: 12px;
            text-align: center;
        }
        .block-container { padding-bottom: 3.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_compliance_footer() -> None:
    st.markdown(f'<div class="compliance-footer">{COMPLIANCE_DISCLOSURE}</div>', unsafe_allow_html=True)


def to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "--", "nan", "None"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def fmt_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.2f}%"


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def safe_date(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT


def classify_fund(name: str) -> str:
    for type_name, keywords in TYPE_KEYWORDS.items():
        if type_name in {"全部", "其他"}:
            continue
        if any(keyword in name for keyword in keywords):
            return type_name
    return "其他"


def benchmark_for_fund(name: str, fund_type: str) -> str:
    if fund_type == "纳斯达克" or "纳斯达克" in name:
        return "纳斯达克100"
    if fund_type == "银行" or "银行" in name:
        return "中证银行"
    if fund_type == "有色金属" or "有色" in name or "金属" in name:
        return "中证有色金属"
    if "科创" in name:
        return "科创50"
    return "沪深300"


def show_notice_once(key: str, message: str, level: str = "warning") -> None:
    """在 Streamlit 界面和终端各提示一次，避免重复刷屏。"""
    if key in UI_NOTICES:
        return
    UI_NOTICES.add(key)
    print(message, flush=True)
    try:
        if level == "error":
            st.error(message)
        elif level == "info":
            st.info(message)
        else:
            st.warning(message)
    except Exception:
        pass


def fetch_nav_with_akshare(code: str) -> pd.DataFrame:
    if importlib.util.find_spec("akshare") is None:
        show_notice_once("akshare_missing", "AkShare 未安装，已自动跳过并改用东方财富接口。", "warning")
        raise ValueError("AkShare 未安装，已跳过")
    try:
        import akshare as ak
    except ImportError as exc:
        show_notice_once("akshare_import_error", f"AkShare 导入失败，已自动跳过：{exc}", "warning")
        raise ValueError(f"AkShare 导入失败：{exc}") from exc

    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势", period="成立来")
    except TypeError:
        df = ak.fund_open_fund_info_em(fund=code, indicator="单位净值走势")
    if df is None or df.empty:
        raise ValueError("AkShare 返回空数据")
    date_col = next((c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"), None)
    nav_col = next((c for c in df.columns if "单位净值" in str(c)), None)
    if nav_col is None:
        nav_col = next((c for c in df.columns if "净值" in str(c) and "日期" not in str(c)), None)
    if date_col is None or nav_col is None:
        raise ValueError(f"AkShare 字段不符合预期：{list(df.columns)}")
    acc_nav_col = next((c for c in df.columns if "累计净值" in str(c)), None)
    out = pd.DataFrame({"date": df[date_col].map(safe_date), "nav": df[nav_col].map(to_float)})
    if acc_nav_col is not None:
        out["acc_nav"] = df[acc_nav_col].map(to_float)
    out = out.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("AkShare 清洗后无有效净值")
    return validate_nav_history(out, code, "AkShare")


def fetch_nav_with_eastmoney(code: str) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=460)).strftime("%Y-%m-%d")
    url = "https://fundf10.eastmoney.com/F10DataApi.aspx"
    frames = []
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 fund-report-streamlit",
            "Referer": f"https://fundf10.eastmoney.com/jjjz_{code}.html",
        }
    )
    page = 1
    max_pages = 1
    while page <= max_pages and page <= 12:
        params = {
            "type": "lsjz",
            "code": code,
            "page": page,
            "per": 49,
            "sdate": start_date,
            "edate": end_date,
        }
        last_error = None
        tables = []
        resp_text = ""
        for attempt in range(1, 4):
            try:
                resp = session.get(url, params=params, timeout=8)
                resp.raise_for_status()
                resp_text = resp.text
                content_match = re.search(r'content:"(.*?)",records:', resp_text, re.S)
                if not content_match:
                    raise ValueError("东方财富返回内容中没有净值表格")
                table_html = content_match.group(1).replace('\\"', '"')
                tables = pd.read_html(StringIO(table_html))
                if not tables:
                    raise ValueError("东方财富净值表格为空")
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2)
                else:
                    show_notice_once(
                        f"eastmoney_retry_failed_{code}_{page}",
                        f"东方财富接口请求失败：{code} 第 {page} 页，已重试 2 次，原因：{exc}",
                        "warning",
                    )
                    raise ValueError(f"东方财富请求失败：{last_error}") from exc
        frames.append(tables[0])
        pages_match = re.search(r"pages:(\d+)", resp_text)
        if pages_match:
            max_pages = min(int(pages_match.group(1)), 12)
        page += 1

    if not frames:
        raise ValueError("东方财富返回空数据")
    df = pd.concat(frames, ignore_index=True)
    date_col = next((c for c in df.columns if "净值日期" in str(c) or "日期" in str(c)), None)
    nav_col = next((c for c in df.columns if "单位净值" in str(c)), None)
    if date_col is None or nav_col is None:
        raise ValueError(f"东方财富字段不符合预期：{list(df.columns)}")
    acc_nav_col = next((c for c in df.columns if "累计净值" in str(c)), None)
    out = pd.DataFrame({"date": df[date_col].map(safe_date), "nav": df[nav_col].map(to_float)})
    if acc_nav_col is not None:
        out["acc_nav"] = df[acc_nav_col].map(to_float)
    out = out.dropna(subset=["date", "nav"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("东方财富清洗后无有效净值")
    return validate_nav_history(out, code, "东方财富")


def validate_nav_history(df: pd.DataFrame, code: str, source: str) -> pd.DataFrame:
    out = df.dropna(subset=["date", "nav"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{source} 清洗后无有效净值")
    invalid_nav = out[(out["nav"] <= 0) | ~np.isfinite(out["nav"])]
    if not invalid_nav.empty:
        raise ValueError(f"{source} 净值存在非正数或无效值")
    if "acc_nav" in out.columns:
        out["acc_nav"] = out["acc_nav"].map(to_float)

    daily_returns = out["nav"].pct_change()
    abnormal = out[daily_returns.abs() > 0.15]
    if not abnormal.empty:
        sample_dates = "、".join(abnormal["date"].dt.strftime("%Y-%m-%d").head(3).tolist())
        show_notice_once(
            f"nav_abnormal_{source}_{code}_{sample_dates}",
            f"{source} 净值合理性预警：{code} 出现单日波动超过 15%（{sample_dates}）。已继续展示，请核对分红、拆分或数据源。",
            "warning",
        )
    return out


def make_mock_nav(code: str) -> pd.DataFrame:
    seed = int(code[-4:])
    rng = random.Random(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=260)
    nav = 1.0 + rng.uniform(-0.15, 0.20)
    rows = []
    for date in dates:
        nav = max(0.2, nav * (1 + rng.normalvariate(0.00025, 0.012)))
        rows.append({"date": date, "nav": nav})
    return pd.DataFrame(rows)


def fetch_nav_history(code: str) -> tuple[pd.DataFrame, str, str]:
    errors = []
    for source, fetcher in [("东方财富", fetch_nav_with_eastmoney), ("AkShare", fetch_nav_with_akshare)]:
        try:
            return fetcher(code), source, ""
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    return make_mock_nav(code), "模拟数据", "；".join(errors)


def fetch_benchmark_history(benchmark_key: str) -> pd.DataFrame:
    benchmark = BENCHMARKS.get(benchmark_key, BENCHMARKS["沪深300"])
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=460)).strftime("%Y%m%d")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": benchmark["secid"],
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56",
        "klt": "101",
        "fqt": "1",
        "beg": start_date,
        "end": end_date,
    }
    resp = requests.get(url, params=params, timeout=8, headers={"User-Agent": "Mozilla/5.0 fund-report-streamlit"})
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise ValueError(f"{benchmark['name']} 基准行情为空")

    rows = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 3:
            continue
        rows.append({"date": safe_date(parts[0]), "nav": to_float(parts[2])})
    out = pd.DataFrame(rows).dropna(subset=["date", "nav"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{benchmark['name']} 基准行情清洗后为空")
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def load_benchmark_history(benchmark_key: str, refresh_token: int = 0) -> tuple[pd.DataFrame, str]:
    _ = refresh_token
    try:
        return fetch_benchmark_history(benchmark_key), benchmark_key
    except Exception as exc:
        if benchmark_key != "沪深300":
            show_notice_once(
                f"benchmark_fallback_{benchmark_key}",
                f"{benchmark_key} 基准行情加载失败，已回退到沪深300：{exc}",
                "warning",
            )
            return fetch_benchmark_history("沪深300"), "沪深300"
        raise


def calc_period_return(df: pd.DataFrame, days: int) -> float:
    if df.empty:
        return float("nan")
    latest = df.iloc[-1]
    target_date = latest["date"] - pd.Timedelta(days=days)
    candidates = df[df["date"] <= target_date]
    if candidates.empty:
        return float("nan")
    base_nav = candidates.iloc[-1]["nav"]
    if base_nav <= 0:
        return float("nan")
    return (latest["nav"] / base_nav - 1) * 100


def nav_on_or_before(df: pd.DataFrame, target_date: Any) -> tuple[float, pd.Timestamp | None]:
    if df.empty:
        return float("nan"), None
    target_ts = pd.to_datetime(target_date)
    candidates = df[df["date"] <= target_ts]
    if candidates.empty:
        return float("nan"), None
    row = candidates.iloc[-1]
    return float(row["nav"]), pd.to_datetime(row["date"])


def update_holding_by_nav(
    holding: dict[str, float],
    nav_df: pd.DataFrame,
    snapshot_date: str,
    auto_update: bool,
) -> dict[str, Any]:
    amount = float(holding.get("amount", 0.0))
    hold_ret = float(holding.get("hold_ret", 0.0))
    cum_profit = float(holding.get("cum_profit", 0.0))
    result = {
        "amount": amount,
        "hold_ret": hold_ret,
        "cum_profit": cum_profit,
        "snapshot_amount": amount,
        "snapshot_hold_ret": hold_ret,
        "snapshot_cum_profit": cum_profit,
        "snapshot_date": snapshot_date,
        "snapshot_nav": float("nan"),
        "snapshot_nav_date": "",
        "holding_auto_updated": False,
        "holding_update_note": "未开启持仓自动更新。",
    }
    if not auto_update or amount <= 0 or nav_df.empty:
        return result

    snapshot_nav, nav_date = nav_on_or_before(nav_df, snapshot_date)
    latest_nav = float(nav_df.iloc[-1]["nav"])
    if math.isnan(snapshot_nav) or snapshot_nav <= 0 or latest_nav <= 0:
        result["holding_update_note"] = "找不到快照日之前的有效净值，已保留原始持仓。"
        return result

    nav_ratio = latest_nav / snapshot_nav
    current_amount = amount * nav_ratio
    cost_basis = amount / (1 + hold_ret / 100.0) if hold_ret > -99.99 else float("nan")
    current_hold_ret = (current_amount / cost_basis - 1) * 100 if cost_basis and not math.isnan(cost_basis) else hold_ret
    current_cum_profit = cum_profit + (current_amount - amount)

    result.update(
        {
            "amount": current_amount,
            "hold_ret": current_hold_ret,
            "cum_profit": current_cum_profit,
            "snapshot_nav": snapshot_nav,
            "snapshot_nav_date": nav_date.strftime("%Y-%m-%d") if nav_date is not None else "",
            "holding_auto_updated": True,
            "holding_update_note": f"按 {result['snapshot_nav_date']} 净值 {snapshot_nav:.4f} 至最新净值 {latest_nav:.4f} 自动更新。",
        }
    )
    return result


def calc_period_return_until(df: pd.DataFrame, end_date: Any, days: int) -> float:
    if df.empty:
        return float("nan")
    end_ts = pd.to_datetime(end_date)
    history = df[df["date"] <= end_ts].copy()
    if history.empty:
        return float("nan")
    latest = history.iloc[-1]
    target_date = latest["date"] - pd.Timedelta(days=days)
    candidates = history[history["date"] <= target_date]
    if candidates.empty:
        return float("nan")
    base_nav = candidates.iloc[-1]["nav"]
    if base_nav <= 0:
        return float("nan")
    return (latest["nav"] / base_nav - 1) * 100


def calc_forward_return(df: pd.DataFrame, start_date: Any, days: int) -> float:
    if df.empty:
        return float("nan")
    start_ts = pd.to_datetime(start_date)
    start_candidates = df[df["date"] <= start_ts]
    end_candidates = df[df["date"] <= start_ts + pd.Timedelta(days=days)]
    if start_candidates.empty or end_candidates.empty:
        return float("nan")
    start_row = start_candidates.iloc[-1]
    end_row = end_candidates.iloc[-1]
    if end_row["date"] <= start_row["date"] or start_row["nav"] <= 0:
        return float("nan")
    return (end_row["nav"] / start_row["nav"] - 1) * 100


def calc_excess_return(fund_df: pd.DataFrame, benchmark_df: pd.DataFrame, days: int) -> float:
    fund_ret = calc_period_return(fund_df, days)
    bench_ret = calc_period_return_until(benchmark_df, fund_df.iloc[-1]["date"], days) if not fund_df.empty else float("nan")
    if math.isnan(fund_ret) or math.isnan(bench_ret):
        return float("nan")
    return fund_ret - bench_ret


def recent_history_until(df: pd.DataFrame, end_date: Any, days: int = 365) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    end_ts = pd.to_datetime(end_date)
    start_ts = end_ts - pd.Timedelta(days=days)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()


def calc_max_drawdown(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    one_year_ago = df.iloc[-1]["date"] - pd.Timedelta(days=365)
    recent = df[df["date"] >= one_year_ago].copy()
    if len(recent) < 2:
        return float("nan")
    rolling_max = recent["nav"].cummax()
    drawdown = recent["nav"] / rolling_max - 1
    return float(drawdown.min() * 100)


def calc_max_drawdown_until(df: pd.DataFrame, end_date: Any, days: int = 365) -> float:
    recent = recent_history_until(df, end_date, days)
    if len(recent) < 2:
        return float("nan")
    rolling_max = recent["nav"].cummax()
    drawdown = recent["nav"] / rolling_max - 1
    return float(drawdown.min() * 100)


def calc_annual_volatility(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    one_year_ago = df.iloc[-1]["date"] - pd.Timedelta(days=365)
    recent = df[df["date"] >= one_year_ago].copy()
    returns = recent["nav"].pct_change().dropna()
    if len(returns) < 30:
        return float("nan")
    return float(returns.std() * np.sqrt(252) * 100)


def calc_annual_volatility_until(df: pd.DataFrame, end_date: Any, days: int = 365) -> float:
    recent = recent_history_until(df, end_date, days)
    returns = recent["nav"].pct_change().dropna() if not recent.empty else pd.Series(dtype="float64")
    if len(returns) < 30:
        return float("nan")
    return float(returns.std() * np.sqrt(252) * 100)


def calc_drawdown_recovery_days(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    one_year_ago = df.iloc[-1]["date"] - pd.Timedelta(days=365)
    recent = df[df["date"] >= one_year_ago].copy().reset_index(drop=True)
    if len(recent) < 2:
        return float("nan")
    rolling_max = recent["nav"].cummax()
    drawdown = recent["nav"] / rolling_max - 1
    trough_idx = int(drawdown.idxmin())
    if drawdown.iloc[trough_idx] >= 0:
        return 0.0
    peak_nav = float(rolling_max.iloc[trough_idx])
    recovered = recent[(recent.index > trough_idx) & (recent["nav"] >= peak_nav)]
    end_date = recovered.iloc[0]["date"] if not recovered.empty else recent.iloc[-1]["date"]
    return float((end_date - recent.iloc[trough_idx]["date"]).days)


def calc_drawdown_recovery_days_until(df: pd.DataFrame, end_date: Any, days: int = 365) -> float:
    recent = recent_history_until(df, end_date, days).reset_index(drop=True)
    if len(recent) < 2:
        return float("nan")
    rolling_max = recent["nav"].cummax()
    drawdown = recent["nav"] / rolling_max - 1
    trough_idx = int(drawdown.idxmin())
    if drawdown.iloc[trough_idx] >= 0:
        return 0.0
    peak_nav = float(rolling_max.iloc[trough_idx])
    recovered = recent[(recent.index > trough_idx) & (recent["nav"] >= peak_nav)]
    recovery_end = recovered.iloc[0]["date"] if not recovered.empty else recent.iloc[-1]["date"]
    return float((recovery_end - recent.iloc[trough_idx]["date"]).days)


def calc_sharpe_until(df: pd.DataFrame, end_date: Any, days: int = 365) -> float:
    recent = recent_history_until(df, end_date, days)
    returns = recent["nav"].pct_change().dropna() if not recent.empty else pd.Series(dtype="float64")
    if len(returns) < 30 or returns.std() == 0:
        return float("nan")
    return float((returns.mean() / returns.std()) * np.sqrt(252))


def build_score_backtest_samples(nav_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> list[dict[str, Any]]:
    if nav_df.empty:
        return []
    df = nav_df.copy().sort_values("date").reset_index(drop=True)
    latest_date = df.iloc[-1]["date"]
    cutoff = latest_date - pd.Timedelta(days=35)
    candidates = df[df["date"] <= cutoff].copy()
    if candidates.empty:
        return []
    candidates["month"] = candidates["date"].dt.to_period("M")
    as_of_dates = candidates.groupby("month").tail(1)["date"].tail(12).tolist()

    samples = []
    for as_of in as_of_dates:
        fund_ret_90 = calc_period_return_until(df, as_of, 90)
        benchmark_ret_90 = calc_period_return_until(benchmark_df, as_of, 90) if benchmark_df is not None and not benchmark_df.empty else float("nan")
        excess_ret_90 = fund_ret_90 - benchmark_ret_90 if not math.isnan(fund_ret_90) and not math.isnan(benchmark_ret_90) else fund_ret_90
        mdd = calc_max_drawdown_until(df, as_of, 180)
        sharpe = calc_sharpe_until(df, as_of, 180)
        volatility = calc_annual_volatility_until(df, as_of, 180)
        recovery_days = calc_drawdown_recovery_days_until(df, as_of, 180)
        calmar = fund_ret_90 / abs(mdd) if not math.isnan(fund_ret_90) and not math.isnan(mdd) and abs(mdd) > 0 else float("nan")
        risk_adjusted = sharpe if not math.isnan(sharpe) else calmar
        stability = -volatility - (recovery_days / 10.0 if not math.isnan(recovery_days) else 0.0) if not math.isnan(volatility) else float("nan")
        target = calc_forward_return(df, as_of, 30)
        if math.isnan(target):
            continue
        samples.append(
            {
                "as_of": pd.to_datetime(as_of).strftime("%Y-%m-%d"),
                "target": target,
                "收益能力": excess_ret_90,
                "风险控制": -abs(mdd) if not math.isnan(mdd) else float("nan"),
                "风险调整收益": risk_adjusted,
                "稳定性": stability,
                "相对排名": excess_ret_90,
            }
        )
    return samples


def calc_sharpe(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    one_year_ago = df.iloc[-1]["date"] - pd.Timedelta(days=365)
    recent = df[df["date"] >= one_year_ago].copy()
    returns = recent["nav"].pct_change().dropna()
    if len(returns) < 30 or returns.std() == 0:
        return float("nan")
    return float((returns.mean() / returns.std()) * np.sqrt(252))


def clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    if math.isnan(value):
        return value
    return min(high, max(low, value))


def percentile_score(values: list[float], higher_better: bool = True) -> list[float]:
    series = pd.Series([to_float(value) for value in values], dtype="float64")
    valid = series.dropna()
    if valid.empty:
        return [5.0 for _ in values]
    if len(valid) == 1:
        return [5.0 if not math.isnan(to_float(value)) else 5.0 for value in values]

    ranks = series.rank(method="average", pct=True, ascending=higher_better)
    scores = ranks.map(lambda x: (float(x) - (1.0 / len(valid))) / (1.0 - (1.0 / len(valid))) * 10.0 if not math.isnan(x) else 5.0)
    return [round(clamp(float(score), 0.0, 10.0), 1) for score in scores.tolist()]


def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    clean = {dimension: max(0.0, float(raw_weights.get(dimension, 0.0))) for dimension in SCORE_DIMENSIONS}
    total = sum(clean.values())
    if total <= 0:
        return DEFAULT_SCORE_WEIGHTS.copy()
    return {dimension: clean[dimension] / total for dimension in SCORE_DIMENSIONS}


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(joined) < 3 or joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return float("nan")
    left_rank = joined.iloc[:, 0].rank(method="average")
    right_rank = joined.iloc[:, 1].rank(method="average")
    left_std = float(left_rank.std(ddof=0))
    right_std = float(right_rank.std(ddof=0))
    if left_std == 0 or right_std == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def calculate_predictive_weights(rows: list[dict[str, Any]], dimension_raw_values: dict[str, list[float]]) -> tuple[dict[str, float], str]:
    _ = dimension_raw_values
    samples = []
    for row in rows:
        for sample in row.get("评分回测样本", []) or []:
            samples.append(sample)

    raw_weights = {}
    if len(samples) >= 12:
        sample_df = pd.DataFrame(samples)
        for dimension in SCORE_DIMENSIONS:
            joined = sample_df[[dimension, "target"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(joined) < 12 or joined[dimension].nunique() < 3 or joined["target"].nunique() < 3:
                raw_weights[dimension] = 0.0
                continue
            ic = spearman_corr(joined[dimension], joined["target"])
            if math.isnan(ic):
                raw_weights[dimension] = 0.0
                continue
            monthly_ics = []
            for _, group in sample_df.groupby("as_of"):
                mini = group[[dimension, "target"]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(mini) >= 3 and mini[dimension].nunique() > 1 and mini["target"].nunique() > 1:
                    mini_ic = spearman_corr(mini[dimension], mini["target"])
                    if not math.isnan(mini_ic):
                        monthly_ics.append(mini_ic)
            ir = abs(ic) if len(monthly_ics) < 2 else abs(float(np.mean(monthly_ics)) / (float(np.std(monthly_ics, ddof=1)) + 1e-9))
            raw_weights[dimension] = abs(float(ic)) * max(ir, 0.01)

    if sum(raw_weights.values()) > 0:
        return normalize_weights(raw_weights), "权重来自历史回测样本 IC/IR"

    return DEFAULT_SCORE_WEIGHTS.copy(), "历史回测样本不足，使用默认审慎权重"


def grade_from_score(total_score: float) -> str:
    if total_score >= 8.5:
        return "五星"
    if total_score >= 7.0:
        return "四星"
    if total_score >= 5.5:
        return "三星"
    if total_score >= 4.0:
        return "二星"
    return "一星"


def apply_composite_scores(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    calmar_values = []
    for row in rows:
        ret_1y = to_float(row.get("近1年收益率"))
        mdd = to_float(row.get("近1年最大回撤"))
        if math.isnan(ret_1y) or math.isnan(mdd) or abs(mdd) <= 0:
            calmar_values.append(float("nan"))
        else:
            calmar_values.append(ret_1y / abs(mdd))

    return_scores = percentile_score(
        [
            to_float(row.get("近1年超额收益率"))
            if not math.isnan(to_float(row.get("近1年超额收益率")))
            else to_float(row.get("近1年收益率"))
            for row in rows
        ],
        higher_better=True,
    )
    risk_scores = percentile_score([abs(to_float(row.get("近1年最大回撤"))) for row in rows], higher_better=False)
    sharpe_scores = percentile_score([to_float(row.get("近1年夏普比率")) for row in rows], higher_better=True)
    calmar_scores = percentile_score(calmar_values, higher_better=True)
    volatility_scores = percentile_score([to_float(row.get("近1年年化波动率")) for row in rows], higher_better=False)
    recovery_scores = percentile_score([to_float(row.get("最大回撤修复天数")) for row in rows], higher_better=False)

    preliminary_values = []
    dimension_raw_values = {dimension: [] for dimension in SCORE_DIMENSIONS}
    for index, row in enumerate(rows):
        risk_adjusted = round((sharpe_scores[index] + calmar_scores[index]) / 2, 1)
        stability = round((volatility_scores[index] + recovery_scores[index]) / 2, 1)
        preliminary = (
            return_scores[index] * DEFAULT_SCORE_WEIGHTS["收益能力"]
            + risk_scores[index] * DEFAULT_SCORE_WEIGHTS["风险控制"]
            + risk_adjusted * DEFAULT_SCORE_WEIGHTS["风险调整收益"]
            + stability * DEFAULT_SCORE_WEIGHTS["稳定性"]
        )
        preliminary_values.append(preliminary)
        dimension_raw_values["收益能力"].append(to_float(row.get("近1年超额收益率")))
        dimension_raw_values["风险控制"].append(-abs(to_float(row.get("近1年最大回撤"))))
        dimension_raw_values["风险调整收益"].append(risk_adjusted)
        dimension_raw_values["稳定性"].append(stability)

    relative_scores = percentile_score(preliminary_values, higher_better=True)
    dimension_raw_values["相对排名"] = preliminary_values
    weights, weight_method = calculate_predictive_weights(rows, dimension_raw_values)

    for index, row in enumerate(rows):
        dimension_scores = {
            "收益能力": return_scores[index],
            "风险控制": risk_scores[index],
            "风险调整收益": round((sharpe_scores[index] + calmar_scores[index]) / 2, 1),
            "稳定性": round((volatility_scores[index] + recovery_scores[index]) / 2, 1),
            "相对排名": relative_scores[index],
        }
        total_score = round(sum(dimension_scores[dimension] * weights[dimension] for dimension in SCORE_DIMENSIONS), 1)
        row["综合评分(0-10)"] = total_score
        row["晨星评级"] = grade_from_score(total_score)
        row["维度分数"] = dimension_scores
        row["评分权重"] = {dimension: round(weights[dimension], 3) for dimension in SCORE_DIMENSIONS}
        row["评分方法"] = f"基金池百分位评分；{weight_method}"
        for dimension, score in dimension_scores.items():
            row[f"{dimension}得分"] = score


def mock_short_advice(name: str, ret_1y: float, mdd: float, hold_ret: float) -> str:
    if not math.isnan(ret_1y) and ret_1y > 15 and abs(mdd) < 20:
        return "表现较强且回撤可控，可继续持有，盈利较多时分批止盈。"
    if not math.isnan(ret_1y) and ret_1y < -10:
        return "近一年偏弱，若持仓亏损扩大，应控制仓位并设止损。"
    if hold_ret > 10:
        return "持有已有收益，可保留底仓，考虑分批止盈锁定利润。"
    return "表现中性，建议继续观察，结合仓位和风险承受力调整。"


def trend_label(ret_1m: float, ret_3m: float) -> str:
    if math.isnan(ret_1m) or math.isnan(ret_3m):
        return "震荡"
    if ret_1m > 3 and ret_3m > 5:
        return "看涨"
    if ret_1m < -3 and ret_3m < -5:
        return "看跌"
    return "震荡"


def mock_detailed_advice(row: dict[str, Any]) -> str:
    ret_1m = to_float(row["近1月收益率"])
    ret_3m = to_float(row["近3月收益率"])
    ret_1y = to_float(row["近1年收益率"])
    mdd = to_float(row["近1年最大回撤"])
    score = to_float(row["综合评分(0-10)"])
    hold_ret = to_float(row["当前持有收益率"])
    trend = trend_label(ret_1m, ret_3m)

    if score >= 8 and hold_ret > 20:
        action = "部分止盈，建议卖出 20%-30%，保留底仓继续跟踪。"
    elif score >= 7:
        action = "继续持有；若回调后趋势未破坏，可小比例逢低加仓。"
    elif score >= 5:
        action = "继续观察，控制仓位，不建议追涨。"
    elif hold_ret < -8 or ret_1m < -8:
        action = "止损清仓或明显减仓，等待趋势企稳后再评估。"
    else:
        action = "降低仓位，优先保留流动性。"

    if score >= 7:
        investor = "平衡型/激进型，建议持有期限 3-12 个月。"
    elif score >= 5:
        investor = "平衡型，建议持有期限 1-6 个月。"
    else:
        investor = "激进型小仓位观察；保守型建议回避或降低仓位。"

    return (
        f"【市场趋势判断】近1月收益率 {fmt_pct(ret_1m)}，近3月收益率 {fmt_pct(ret_3m)}，"
        f"未来1-4周倾向于{trend}。\n\n"
        f"【具体操作建议】{action}\n\n"
        f"【风险提示】1. 最大回撤 {fmt_pct(mdd)}，波动风险不可忽视；"
        f"2. 近1年收益率 {fmt_pct(ret_1y)} 不代表未来收益；3. 主题基金需关注估值、政策和汇率等变化。\n\n"
        f"【适合人群】{investor}"
    )


def ensure_ai_disclaimer(text: str) -> str:
    clean = str(text or "").strip()
    if AI_DISCLAIMER in clean:
        return clean
    return f"{clean}\n\n{AI_DISCLAIMER}" if clean else AI_DISCLAIMER


def advice_numbers_are_consistent(text: str, row: dict[str, Any], tolerance: float = 0.35) -> bool:
    checks = [
        (r"近\s*1\s*月\s*收益率?\s*[为是:]?\s*([+-]?\d+(?:\.\d+)?)\s*%", "近1月收益率"),
        (r"近\s*3\s*月\s*收益率?\s*[为是:]?\s*([+-]?\d+(?:\.\d+)?)\s*%", "近3月收益率"),
        (r"近\s*1\s*年\s*收益率?\s*[为是:]?\s*([+-]?\d+(?:\.\d+)?)\s*%", "近1年收益率"),
        (r"近\s*1\s*年\s*超额收益率?\s*[为是:]?\s*([+-]?\d+(?:\.\d+)?)\s*%", "近1年超额收益率"),
        (r"超额收益率?\s*[为是:]?\s*([+-]?\d+(?:\.\d+)?)\s*%", "近1年超额收益率"),
        (r"近\s*1\s*年\s*最大回撤\s*[为是:]?\s*([+-]?\d+(?:\.\d+)?)\s*%", "近1年最大回撤"),
        (r"最大回撤\s*[为是:]?\s*([+-]?\d+(?:\.\d+)?)\s*%", "近1年最大回撤"),
        (r"年化波动率?\s*[为是:]?\s*([+-]?\d+(?:\.\d+)?)\s*%", "近1年年化波动率"),
    ]
    for pattern, field in checks:
        matches = re.findall(pattern, text)
        if not matches:
            continue
        actual = to_float(row.get(field))
        if math.isnan(actual):
            continue
        if not any(abs(float(match) - actual) <= tolerance for match in matches):
            return False
    return True


def validated_advice(short_text: str, detail_text: str, row: dict[str, Any], fallback_short: str, fallback_detail: str) -> tuple[str, str]:
    merged = f"{short_text}\n{detail_text}"
    if not advice_numbers_are_consistent(merged, row):
        show_notice_once(
            f"ai_advice_metric_mismatch_{row['代码']}",
            f"{row['基金名称']} 的 AI 建议引用数据与表格不一致，已改用本地校验文案。",
            "warning",
        )
        return fallback_short, ensure_ai_disclaimer(fallback_detail)
    return short_text or fallback_short, ensure_ai_disclaimer(detail_text or fallback_detail)


def parse_qwen_advice_pair(text: str) -> tuple[str, str]:
    """解析一次 AI 调用返回的简短和详细建议。"""
    short_match = re.search(r"【简短建议】\s*(.*?)(?=【详细建议】|$)", text, re.S)
    detail_match = re.search(r"【详细建议】\s*(.*)$", text, re.S)
    short_text = short_match.group(1).strip() if short_match else ""
    detail_text = detail_match.group(1).strip() if detail_match else ""
    return short_text, detail_text


def get_dashscope_api_key() -> str:
    session_key = st.session_state.get("dashscope_api_key", "").strip()
    if session_key:
        return session_key
    env_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("DASHSCOPE_API_KEY", "")).strip()
    except Exception:
        return ""


def call_qwen_advice_pair(row: dict[str, Any]) -> tuple[str, str]:
    """一次通义千问调用同时生成简短建议和详细建议，减少 API 调用次数。"""
    api_key = get_dashscope_api_key()
    mock_short = mock_short_advice(
        row["基金名称"],
        to_float(row["近1年收益率"]),
        to_float(row["近1年最大回撤"]),
        to_float(row["当前持有收益率"]),
    )
    mock_detail = mock_detailed_advice(row)
    if not api_key:
        return mock_short, ensure_ai_disclaimer(mock_detail)

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=10.0,
            max_retries=0,
        )
        model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        prompt = f"""
请基于以下基金数据，输出中文建议，不承诺收益。
基金名称：{row["基金名称"]}
代码：{row["代码"]}
近1月收益率：{fmt_pct(to_float(row["近1月收益率"]))}
近3月收益率：{fmt_pct(to_float(row["近3月收益率"]))}
近1年收益率：{fmt_pct(to_float(row["近1年收益率"]))}
近1年基准收益率（{row.get("比较基准", "沪深300")}）：{fmt_pct(to_float(row.get("近1年基准收益率")))}
近1年超额收益率：{fmt_pct(to_float(row.get("近1年超额收益率")))}
近1年最大回撤：{fmt_pct(to_float(row["近1年最大回撤"]))}
近1年年化波动率：{fmt_pct(to_float(row.get("近1年年化波动率")))}
最大回撤修复天数：{fmt_num(to_float(row.get("最大回撤修复天数")), 0)} 天
夏普比率：{fmt_num(to_float(row["近1年夏普比率"]))}
综合评分：{row["综合评分(0-10)"]}
晨星评级：{row["晨星评级"]}
当前持有收益率：{row["当前持有收益率"]:.2f}%

请严格使用以下格式：
【简短建议】
用 50 字以内给出继续持有、止盈或止损建议。

【详细建议】
控制在 300 字以内，必须包含四段：
【市场趋势判断】判断未来1-4周走势：看涨/震荡/看跌。
【具体操作建议】明确给出继续持有、部分止盈比例、逢低加仓或止损清仓。
【风险提示】列出2-3个当前主要风险点。
【适合人群】说明保守型/平衡型/激进型及持有期限建议。
禁止编造或改写上述数值；如引用收益率、回撤、波动率，必须与输入数据完全一致。
""".strip()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是谨慎、客观的基金组合分析助手。不要承诺收益。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        text = response.choices[0].message.content.strip()
        if text:
            short_text, detail_text = parse_qwen_advice_pair(text)
            return validated_advice(short_text, detail_text, row, mock_short, mock_detail)
    except Exception as exc:
        show_notice_once("qwen_call_failed", f"通义千问调用失败，已使用本地 mock 建议：{exc}", "warning")

    return mock_short, ensure_ai_disclaimer(mock_detail)


def analyze_fund(
    fund: dict[str, str],
    holdings: dict[str, dict[str, float]],
    refresh_token: int = 0,
    snapshot_date: str = "",
    auto_update_holdings: bool = False,
) -> dict[str, Any]:
    code = fund["code"]
    name = fund["name"]
    fund_type = classify_fund(name)
    benchmark_key = benchmark_for_fund(name, fund_type)
    nav_df, source, error_msg = fetch_nav_history(code)
    benchmark_df, benchmark_used = load_benchmark_history(benchmark_key, refresh_token)
    latest = nav_df.iloc[-1]
    ret_1m = calc_period_return(nav_df, 30)
    ret_3m = calc_period_return(nav_df, 90)
    ret_1y = calc_period_return(nav_df, 365)
    benchmark_ret_1y = calc_period_return_until(benchmark_df, latest["date"], 365)
    excess_ret_1m = calc_excess_return(nav_df, benchmark_df, 30)
    excess_ret_1y = calc_excess_return(nav_df, benchmark_df, 365)
    mdd = calc_max_drawdown(nav_df)
    sharpe = calc_sharpe(nav_df)
    volatility = calc_annual_volatility(nav_df)
    recovery_days = calc_drawdown_recovery_days(nav_df)
    holding = holdings.get(code, {"amount": 0.0, "hold_ret": 0.0, "cum_profit": 0.0})
    updated_holding = update_holding_by_nav(holding, nav_df, snapshot_date, auto_update_holdings)
    return {
        "基金名称": name,
        "代码": code,
        "基金类型": fund_type,
        "比较基准": benchmark_used,
        "最新净值日期": latest["date"].strftime("%Y-%m-%d"),
        "单位净值": float(latest["nav"]),
        "近1月收益率": ret_1m,
        "近3月收益率": ret_3m,
        "近1年收益率": ret_1y,
        "近1年基准收益率": benchmark_ret_1y,
        "近1月超额收益率": excess_ret_1m,
        "近1年超额收益率": excess_ret_1y,
        "近1年最大回撤": mdd,
        "近1年夏普比率": sharpe,
        "近1年年化波动率": volatility,
        "最大回撤修复天数": recovery_days,
        "当前持有金额": float(updated_holding["amount"]),
        "当前持有收益率": float(updated_holding["hold_ret"]),
        "累计收益": float(updated_holding["cum_profit"]),
        "快照持有金额": float(updated_holding["snapshot_amount"]),
        "快照持有收益率": float(updated_holding["snapshot_hold_ret"]),
        "快照累计收益": float(updated_holding["snapshot_cum_profit"]),
        "持仓快照日期": updated_holding["snapshot_date"],
        "快照净值日期": updated_holding["snapshot_nav_date"],
        "快照单位净值": float(updated_holding["snapshot_nav"]),
        "持仓已自动更新": bool(updated_holding["holding_auto_updated"]),
        "持仓更新说明": updated_holding["holding_update_note"],
        "数据来源": source,
        "错误信息": error_msg,
        "评分回测样本": build_score_backtest_samples(nav_df, benchmark_df),
    }


def analyze_fund_with_mock(
    fund: dict[str, str],
    nav_df: pd.DataFrame,
    error_msg: str,
    holdings: dict[str, dict[str, float]],
    snapshot_date: str = "",
    auto_update_holdings: bool = False,
) -> dict[str, Any]:
    code = fund["code"]
    name = fund["name"]
    fund_type = classify_fund(name)
    latest = nav_df.iloc[-1]
    ret_1m = calc_period_return(nav_df, 30)
    ret_3m = calc_period_return(nav_df, 90)
    ret_1y = calc_period_return(nav_df, 365)
    mdd = calc_max_drawdown(nav_df)
    sharpe = calc_sharpe(nav_df)
    volatility = calc_annual_volatility(nav_df)
    recovery_days = calc_drawdown_recovery_days(nav_df)
    holding = holdings.get(code, {"amount": 0.0, "hold_ret": 0.0, "cum_profit": 0.0})
    updated_holding = update_holding_by_nav(holding, nav_df, snapshot_date, auto_update_holdings)
    return {
        "基金名称": name,
        "代码": code,
        "基金类型": fund_type,
        "比较基准": "沪深300",
        "最新净值日期": latest["date"].strftime("%Y-%m-%d"),
        "单位净值": float(latest["nav"]),
        "近1月收益率": ret_1m,
        "近3月收益率": ret_3m,
        "近1年收益率": ret_1y,
        "近1年基准收益率": float("nan"),
        "近1月超额收益率": float("nan"),
        "近1年超额收益率": float("nan"),
        "近1年最大回撤": mdd,
        "近1年夏普比率": sharpe,
        "近1年年化波动率": volatility,
        "最大回撤修复天数": recovery_days,
        "当前持有金额": float(updated_holding["amount"]),
        "当前持有收益率": float(updated_holding["hold_ret"]),
        "累计收益": float(updated_holding["cum_profit"]),
        "快照持有金额": float(updated_holding["snapshot_amount"]),
        "快照持有收益率": float(updated_holding["snapshot_hold_ret"]),
        "快照累计收益": float(updated_holding["snapshot_cum_profit"]),
        "持仓快照日期": updated_holding["snapshot_date"],
        "快照净值日期": updated_holding["snapshot_nav_date"],
        "快照单位净值": float(updated_holding["snapshot_nav"]),
        "持仓已自动更新": bool(updated_holding["holding_auto_updated"]),
        "持仓更新说明": updated_holding["holding_update_note"],
        "数据来源": "模拟数据",
        "错误信息": error_msg,
        "评分回测样本": [],
    }


@st.cache_data(ttl=86400, show_spinner="数据加载中，请稍候...")
def load_data(
    refresh_token: int = 0,
    holdings_json: str = "",
    snapshot_date: str = "",
    auto_update_holdings: bool = False,
) -> list[dict[str, Any]]:
    try:
        holdings_list = normalize_holdings(json.loads(holdings_json or "[]"))
    except Exception:
        holdings_list = default_holdings()
    holdings = holdings_to_map(holdings_list)
    rows = []
    for holding_item in holdings_list:
        fund = {"code": holding_item["code"], "name": holding_item["name"]}
        try:
            rows.append(analyze_fund(fund, holdings, refresh_token, snapshot_date, auto_update_holdings))
        except Exception as exc:
            rows.append(
                analyze_fund_with_mock(
                    fund,
                    make_mock_nav(fund["code"]),
                    str(exc),
                    holdings,
                    snapshot_date,
                    auto_update_holdings,
                )
            )

    apply_composite_scores(rows)
    for row in rows:
        short_advice, detailed_advice = call_qwen_advice_pair(row)
        row["AI建议"] = short_advice
        row["详细AI建议"] = detailed_advice
    return rows


def filter_rows(rows: list[dict[str, Any]], score_range: tuple[float, float], fund_type: str) -> list[dict[str, Any]]:
    low, high = score_range
    filtered = [row for row in rows if low <= float(row["综合评分(0-10)"]) <= high]
    if fund_type != "全部":
        filtered = [row for row in filtered if row["基金类型"] == fund_type]
    return filtered


def apply_manual_overrides(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """应用当前会话中的手动覆盖数据，并重新计算评分。"""
    adjusted_rows = copy.deepcopy(rows)
    overrides = st.session_state.get("manual_overrides", {})
    if not overrides:
        return adjusted_rows

    for row in adjusted_rows:
        code = str(row["代码"]).zfill(6)
        if code not in overrides:
            continue
        row.update(overrides[code])
        row["数据来源"] = "手动覆盖"
        row["评分回测样本"] = []
        row["持仓更新说明"] = "手动覆盖后未重新推算持仓快照。"

    apply_composite_scores(adjusted_rows)
    for row in adjusted_rows:
        code = str(row["代码"]).zfill(6)
        if code in overrides:
            row["AI建议"] = mock_short_advice(
                row["基金名称"],
                to_float(row["近1年收益率"]),
                to_float(row["近1年最大回撤"]),
                to_float(row["当前持有收益率"]),
            )
            row["详细AI建议"] = ensure_ai_disclaimer(mock_detailed_advice(row))
    return adjusted_rows


def safe_number_value(value: Any, default: float = 0.0) -> float:
    number = to_float(value)
    if math.isnan(number) or math.isinf(number):
        return default
    return float(number)


def render_manual_override_panel(rows: list[dict[str, Any]]) -> None:
    """侧边栏手动覆盖基金指标，用于接口失败或临时修正数据。"""
    if "manual_overrides" not in st.session_state:
        st.session_state.manual_overrides = {}

    with st.sidebar.expander("手动覆盖基金数据", expanded=False):
        options = {f"{row['基金名称']}（{str(row['代码']).zfill(6)}）": row for row in rows}
        label = st.selectbox("选择基金", list(options.keys()), key="override_fund_select")
        base = options[label]
        code = str(base["代码"]).zfill(6)
        current = {**base, **st.session_state.manual_overrides.get(code, {})}

        latest_nav = st.number_input("单位净值", value=safe_number_value(current.get("单位净值"), 1.0), step=0.0001, format="%.4f")
        ret_1m = st.number_input("近1月收益率(%)", value=safe_number_value(current.get("近1月收益率")), step=0.1, format="%.2f")
        ret_3m = st.number_input("近3月收益率(%)", value=safe_number_value(current.get("近3月收益率")), step=0.1, format="%.2f")
        ret_1y = st.number_input("近1年收益率(%)", value=safe_number_value(current.get("近1年收益率")), step=0.1, format="%.2f")
        mdd = st.number_input("近1年最大回撤(%)", value=safe_number_value(current.get("近1年最大回撤")), step=0.1, format="%.2f")
        sharpe = st.number_input("近1年夏普比率", value=safe_number_value(current.get("近1年夏普比率")), step=0.1, format="%.2f")

        c1, c2 = st.columns(2)
        if c1.button("应用覆盖", use_container_width=True):
            st.session_state.manual_overrides[code] = {
                "单位净值": latest_nav,
                "近1月收益率": ret_1m,
                "近3月收益率": ret_3m,
                "近1年收益率": ret_1y,
                "近1年最大回撤": mdd,
                "近1年夏普比率": sharpe,
            }
            st.rerun()
        if c2.button("清除覆盖", use_container_width=True):
            st.session_state.manual_overrides.pop(code, None)
            st.rerun()

        if st.session_state.manual_overrides:
            st.caption(f"当前已覆盖 {len(st.session_state.manual_overrides)} 只基金。")


def render_holdings_editor() -> None:
    """侧边栏编辑当前会话持仓数据。"""
    with st.sidebar.expander("编辑我的持仓", expanded=False):
        holdings = normalize_holdings(st.session_state.holdings)
        rows = []
        for item in holdings:
            rows.append(
                {
                    "基金代码": item["code"],
                    "基金名称": item["name"],
                    "持有金额(元)": float(item.get("amount", 0.0)),
                    "持有收益率(%)": float(item.get("hold_ret", 0.0)),
                    "累计收益(元)": float(item.get("cum_profit", 0.0)),
                }
            )

        st.caption("可新增、删除、修改基金代码和名称。基金代码建议填写 6 位数字。")
        edited_df = st.data_editor(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "基金代码": st.column_config.TextColumn(required=True, help="例如 006887"),
                "基金名称": st.column_config.TextColumn(required=True),
                "持有金额(元)": st.column_config.NumberColumn(format="%.2f", step=10.0),
                "持有收益率(%)": st.column_config.NumberColumn(format="%.2f", step=0.1),
                "累计收益(元)": st.column_config.NumberColumn(format="%.2f", step=10.0),
            },
            key="holdings_editor",
        )

        c1, c2 = st.columns(2)
        if c1.button("保存持仓修改", use_container_width=True):
            updated_holdings = []
            invalid_rows = []
            seen_codes = set()
            for _, row in edited_df.iterrows():
                code = normalize_fund_code(row.get("基金代码"))
                name = str(row.get("基金名称") or "").strip()
                if not code or not name:
                    invalid_rows.append(str(row.get("基金代码") or "空代码"))
                    continue
                if code in seen_codes:
                    invalid_rows.append(f"{code}（重复）")
                    continue
                seen_codes.add(code)
                updated_holdings.append(
                    {
                        "code": code,
                        "name": name,
                        "amount": float(to_float(row["持有金额(元)"])),
                        "hold_ret": float(to_float(row["持有收益率(%)"])),
                        "cum_profit": float(to_float(row["累计收益(元)"])),
                    }
                )
            if invalid_rows:
                st.warning("以下行未保存，请检查代码/名称或重复代码：" + "、".join(invalid_rows))
                return
            if not updated_holdings:
                st.warning("至少保留一只基金。")
                return
            st.session_state.holdings = normalize_holdings(updated_holdings)
            st.cache_data.clear()
            st.session_state.refresh_token += 1
            st.success("持仓已保存，正在刷新分析结果。")
            st.rerun()

        if c2.button("恢复示例", use_container_width=True):
            st.session_state.holdings = default_holdings()
            st.cache_data.clear()
            st.session_state.refresh_token += 1
            st.rerun()

        st.caption("持仓只保存在当前浏览器会话中。保存后会按你当前组合重新抓取数据、评分并生成 AI 建议。")


def build_table_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    table_df = df[
        [
            "基金名称",
            "代码",
            "基金类型",
            "当前持有金额",
            "当前持有收益率",
            "快照持有金额",
            "持仓快照日期",
            "近1月收益率",
            "近1年收益率",
            "近1年超额收益率",
            "综合评分(0-10)",
            "晨星评级",
            "数据来源",
            "AI建议",
        ]
    ].copy()
    table_df["代码"] = table_df["代码"].astype(str).str.zfill(6)
    for col in ["当前持有金额", "当前持有收益率", "快照持有金额", "近1月收益率", "近1年收益率", "近1年超额收益率", "综合评分(0-10)"]:
        table_df[col] = table_df[col].map(lambda x: round(to_float(x), 2))
    return table_df


def highlight_mock_rows(row: pd.Series) -> list[str]:
    if row.get("数据来源") == "模拟数据":
        return ["background-color: #7f1d1d; color: #fee2e2;" for _ in row]
    return ["" for _ in row]


def portfolio_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_amount": sum(float(row["当前持有金额"]) for row in rows),
        "total_profit": sum(float(row["累计收益"]) for row in rows),
        "best": max(rows, key=lambda item: item["综合评分(0-10)"]),
        "worst": min(rows, key=lambda item: item["综合评分(0-10)"]),
        "lowest_risk": max(rows, key=lambda item: item["风险控制得分"]),
        "avg_score": sum(float(row["综合评分(0-10)"]) for row in rows) / len(rows),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def bar_chart(rows: list[dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(rows)
    df["基金标签"] = df["基金名称"] + " (" + df["代码"].astype(str).str.zfill(6) + ")"
    df["回撤绝对值"] = df["近1年最大回撤"].map(lambda x: abs(to_float(x)))
    fig = px.bar(
        df,
        x="基金标签",
        y="近1年收益率",
        color="回撤绝对值",
        color_continuous_scale="RdYlGn_r",
        hover_data=["近1月收益率", "近3月收益率", "近1年最大回撤", "综合评分(0-10)", "晨星评级"],
        title="收益率对比柱状图",
    )
    fig.update_layout(height=460, xaxis_tickangle=-35, margin=dict(l=30, r=20, t=60, b=120))
    return fig


def scatter_chart(rows: list[dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="近1年最大回撤",
        y="近1年收益率",
        size="综合评分(0-10)",
        color="晨星评级",
        text="基金名称",
        hover_data=["近1月收益率", "近3月收益率", "收益能力得分", "风险控制得分"],
        title="收益-回撤散点图",
    )
    fig.add_vline(x=-15, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=20, line_dash="dash", line_color="#94a3b8")
    fig.add_annotation(x=-7, y=32, text="高收益低回撤", showarrow=False)
    fig.add_annotation(x=-26, y=32, text="高收益高回撤", showarrow=False)
    fig.add_annotation(x=-7, y=4, text="低收益低回撤", showarrow=False)
    fig.add_annotation(x=-26, y=4, text="低收益高回撤", showarrow=False)
    fig.update_traces(textposition="top center")
    fig.update_layout(height=460, margin=dict(l=30, r=20, t=60, b=50))
    return fig


def radar_chart(row: dict[str, Any]) -> go.Figure:
    categories = ["收益能力", "风险控制", "风险调整收益", "稳定性", "相对排名"]
    values = [row["维度分数"][category] for category in categories]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=row["基金名称"],
        )
    )
    fig.update_layout(
        title=f"{row['基金名称']} 五维评分雷达图",
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        height=520,
        margin=dict(l=30, r=30, t=70, b=40),
    )
    return fig


def selected_funds_radar_chart(rows: list[dict[str, Any]]) -> go.Figure:
    """展示前 3 名和后 3 名基金的雷达图，避免过多曲线叠加。"""
    categories = ["收益能力", "风险控制", "风险调整收益", "稳定性", "相对排名"]
    fig = go.Figure()
    sorted_rows = sorted(rows, key=lambda item: item["综合评分(0-10)"], reverse=True)
    selected_rows = sorted_rows[:3] + [row for row in sorted_rows[-3:] if row not in sorted_rows[:3]]
    for row in selected_rows:
        values = [row["维度分数"][category] for category in categories]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"{row['基金名称']} {row['综合评分(0-10)']}",
                opacity=0.58,
            )
        )
    fig.update_layout(
        title="综合评分雷达图（前3名与后3名）",
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        height=650,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


@st.cache_data(ttl=86400, show_spinner="正在加载回测净值...")
def load_nav_for_backtest(code: str, refresh_token: int = 0) -> tuple[pd.DataFrame, str, str]:
    _ = refresh_token
    return fetch_nav_history(code)


def simulate_monthly_investment(
    nav_df: pd.DataFrame,
    months: int,
    monthly_amount: float,
    subscription_fee_rate: float = 0.0,
    redemption_fee_rate: float = 0.0,
    dividend_mode: str = "红利再投",
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """模拟过去 N 个月每月定投固定金额，并按最新净值估算当前收益。"""
    if nav_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    df = nav_df.copy().sort_values("date").reset_index(drop=True)
    price_col = "nav"
    dividend_note = "现金分红：按单位净值口径估算，未额外计入历史现金分红。"
    if dividend_mode == "红利再投" and "acc_nav" in df.columns and df["acc_nav"].dropna().nunique() > 1:
        price_col = "acc_nav"
        dividend_note = "红利再投：使用累计净值口径近似估算分红再投资效果。"
    elif dividend_mode == "红利再投":
        dividend_note = "红利再投：当前数据缺少累计净值，已回退为单位净值口径。"

    latest_date = df.iloc[-1]["date"]
    latest_nav = float(df.iloc[-1][price_col])
    start_date = latest_date - pd.DateOffset(months=months)
    recent = df[df["date"] >= start_date].copy()
    if recent.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    recent["month"] = recent["date"].dt.to_period("M")
    purchases = recent.groupby("month", as_index=False).first().tail(months).copy()
    purchases["投入金额"] = float(monthly_amount)
    purchases["申购费"] = purchases["投入金额"] * max(0.0, subscription_fee_rate) / 100.0
    purchases["净申购金额"] = purchases["投入金额"] - purchases["申购费"]
    purchases["买入份额"] = purchases["净申购金额"] / purchases[price_col]
    purchases["当前价值"] = purchases["买入份额"] * latest_nav
    purchases["当前盈亏"] = purchases["当前价值"] - purchases["投入金额"]

    first_purchase_date = purchases.iloc[0]["date"]
    curve = df[df["date"] >= first_purchase_date].copy()
    invested_values = []
    market_values = []
    for _, day in curve.iterrows():
        active = purchases[purchases["date"] <= day["date"]]
        invested = float(active["投入金额"].sum())
        shares = float(active["买入份额"].sum())
        invested_values.append(invested)
        market_values.append(shares * float(day[price_col]))
    curve["累计投入"] = invested_values
    curve["组合当前价值"] = market_values
    curve["收益率"] = np.where(curve["累计投入"] > 0, curve["组合当前价值"] / curve["累计投入"] - 1, 0.0)

    total_invested = float(purchases["投入金额"].sum())
    total_subscription_fee = float(purchases["申购费"].sum())
    total_shares = float(purchases["买入份额"].sum())
    gross_value = total_shares * latest_nav
    redemption_fee = gross_value * max(0.0, redemption_fee_rate) / 100.0
    current_value = gross_value - redemption_fee
    profit = current_value - total_invested
    profit_rate = profit / total_invested * 100 if total_invested > 0 else 0.0
    summary = {
        "total_invested": total_invested,
        "total_subscription_fee": total_subscription_fee,
        "redemption_fee": redemption_fee,
        "gross_value": gross_value,
        "current_value": current_value,
        "profit": profit,
        "profit_rate": profit_rate,
        "latest_nav": latest_nav,
        "price_col": price_col,
        "dividend_note": dividend_note,
    }
    return summary, purchases, curve


def render_backtest_simulator(rows: list[dict[str, Any]], selected_row: dict[str, Any]) -> None:
    st.subheader("定投回测模拟器")
    options = {f"{row['基金名称']}（{str(row['代码']).zfill(6)}）": row for row in rows}
    labels = list(options.keys())
    default_label = next(
        (label for label, row in options.items() if row["代码"] == selected_row["代码"]),
        labels[0],
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    fund_label = c1.selectbox("选择回测基金", labels, index=labels.index(default_label), key="backtest_fund")
    months = c2.number_input("回测月份", min_value=1, max_value=24, value=3, step=1)
    monthly_amount = c3.number_input("每月定投金额", min_value=10.0, max_value=100000.0, value=100.0, step=10.0)
    f1, f2, f3 = st.columns([1, 1, 1])
    subscription_fee_rate = f1.number_input("申购费率(%)", min_value=0.0, max_value=5.0, value=0.0, step=0.05, format="%.2f")
    redemption_fee_rate = f2.number_input("赎回费率(%)", min_value=0.0, max_value=5.0, value=0.0, step=0.05, format="%.2f")
    dividend_mode = f3.selectbox("分红方式", ["红利再投", "现金分红"], index=0)

    fund = options[fund_label]
    try:
        nav_df, source, err = load_nav_for_backtest(str(fund["代码"]).zfill(6), st.session_state.refresh_token)
        summary, purchases, curve = simulate_monthly_investment(
            nav_df,
            int(months),
            float(monthly_amount),
            float(subscription_fee_rate),
            float(redemption_fee_rate),
            dividend_mode,
        )
    except Exception as exc:
        st.warning(f"回测数据加载失败：{exc}")
        return

    if not summary:
        st.warning("净值数据不足，无法完成回测。")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("累计投入", f"{summary['total_invested']:.2f}")
    m2.metric("当前价值", f"{summary['current_value']:.2f}")
    m3.metric("当前盈亏", f"{summary['profit']:.2f}")
    m4.metric("收益率", f"{summary['profit_rate']:.2f}%")
    st.caption(
        f"回测数据来源：{source}。假设每月第一个可用净值日买入，已扣除申购费 {subscription_fee_rate:.2f}% "
        f"和赎回费 {redemption_fee_rate:.2f}%。{summary['dividend_note']} 历史回测不代表未来表现。"
    )
    if err:
        st.caption(f"数据提示：{err}")
    st.caption(f"费用估算：申购费合计 {summary['total_subscription_fee']:.2f}，期末赎回费 {summary['redemption_fee']:.2f}。")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["累计投入"], mode="lines", name="累计投入"))
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["组合当前价值"], mode="lines", name="组合价值"))
    fig.update_layout(height=380, title="定投组合价值曲线", margin=dict(l=30, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

    purchase_cols = ["date", "nav", "投入金额", "申购费", "净申购金额", "买入份额", "当前价值", "当前盈亏"]
    if "acc_nav" in purchases.columns:
        purchase_cols.insert(2, "acc_nav")
    purchase_table = purchases[purchase_cols].copy()
    purchase_table["date"] = purchase_table["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(purchase_table, use_container_width=True, hide_index=True)


def selected_row_from_table(event: Any, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        selected = event.selection.rows
    except Exception:
        selected = []
    if not selected:
        return None
    idx = selected[0]
    if 0 <= idx < len(rows):
        return rows[idx]
    return None


def build_markdown_report(rows: list[dict[str, Any]]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(rows)
    sorted_df = df.sort_values("综合评分(0-10)", ascending=False)
    top3 = sorted_df.head(3)
    bottom3 = sorted_df.tail(3).sort_values("综合评分(0-10)")

    table_df = df[
        [
            "基金名称",
            "代码",
            "比较基准",
            "近1月收益率",
            "近1年收益率",
            "近1年超额收益率",
            "近1年最大回撤",
            "综合评分(0-10)",
            "晨星评级",
            "数据来源",
            "AI建议",
        ]
    ].copy()
    for col in ["近1月收益率", "近1年收益率", "近1年超额收益率", "近1年最大回撤"]:
        table_df[col] = table_df[col].map(lambda x: fmt_pct(to_float(x)))

    detail_df = df[
        [
            "基金名称",
            "代码",
            "最新净值日期",
            "单位净值",
            "持仓快照日期",
            "快照净值日期",
            "快照单位净值",
            "快照持有金额",
            "快照持有收益率",
            "近3月收益率",
            "近1年基准收益率",
            "近1月超额收益率",
            "近1年超额收益率",
            "近1年夏普比率",
            "近1年年化波动率",
            "最大回撤修复天数",
            "当前持有金额",
            "当前持有收益率",
            "累计收益",
            "收益能力得分",
            "风险控制得分",
            "风险调整收益得分",
            "稳定性得分",
            "相对排名得分",
            "数据来源",
        ]
    ].copy()
    detail_df["单位净值"] = detail_df["单位净值"].map(lambda x: fmt_num(to_float(x), 4))
    detail_df["快照单位净值"] = detail_df["快照单位净值"].map(lambda x: fmt_num(to_float(x), 4))
    detail_df["快照持有金额"] = detail_df["快照持有金额"].map(lambda x: f"{to_float(x):.2f}")
    detail_df["快照持有收益率"] = detail_df["快照持有收益率"].map(lambda x: f"{to_float(x):.2f}%")
    detail_df["近3月收益率"] = detail_df["近3月收益率"].map(lambda x: fmt_pct(to_float(x)))
    for col in ["近1年基准收益率", "近1月超额收益率", "近1年超额收益率", "近1年年化波动率"]:
        detail_df[col] = detail_df[col].map(lambda x: fmt_pct(to_float(x)))
    detail_df["近1年夏普比率"] = detail_df["近1年夏普比率"].map(lambda x: fmt_num(to_float(x), 2))
    detail_df["最大回撤修复天数"] = detail_df["最大回撤修复天数"].map(lambda x: fmt_num(to_float(x), 0))
    detail_df["当前持有金额"] = detail_df["当前持有金额"].map(lambda x: f"{to_float(x):.2f}")
    detail_df["当前持有收益率"] = detail_df["当前持有收益率"].map(lambda x: f"{to_float(x):.2f}%")
    detail_df["累计收益"] = detail_df["累计收益"].map(lambda x: f"{to_float(x):.2f}")
    for col in ["收益能力得分", "风险控制得分", "风险调整收益得分", "稳定性得分", "相对排名得分"]:
        detail_df[col] = detail_df[col].map(lambda x: fmt_num(to_float(x), 1))

    top_text = "、".join([f"{r['基金名称']}（{r['综合评分(0-10)']}分，{r['晨星评级']}）" for _, r in top3.iterrows()])
    bottom_text = "、".join([f"{r['基金名称']}（{r['综合评分(0-10)']}分，{r['晨星评级']}）" for _, r in bottom3.iterrows()])
    warning_df = df[df["数据来源"] == "模拟数据"]
    warning_text = ""
    if not warning_df.empty:
        warning_text = "\n> 注意：以下基金使用了模拟数据，仅用于流程验证：" + "、".join(warning_df["基金名称"].tolist()) + "\n"

    return f"""# 基金量化分析报告

生成时间：{generated_at}

> 免责声明：{COMPLIANCE_DISCLOSURE}
{warning_text}
## 核心评分与建议

{table_df.to_markdown(index=False)}

## 持仓与数据明细

{detail_df.to_markdown(index=False)}

综合评分最高的 3 只基金：{top_text}。

综合评分最低的 3 只基金：{bottom_text}。

多维度评分说明：各指标均在当前基金池内转换为 0-10 百分位分。收益能力优先使用基金近 1 年收益率减同期比较基准收益率后的超额收益；风险控制使用最大回撤百分位；风险调整收益由夏普比率和卡玛比率百分位共同决定；稳定性使用年化波动率和最大回撤修复天数；相对排名依据维度预评分百分位给分。

权重说明：系统会根据历史样本 IC/IR 估计维度权重，历史样本不足时回退默认权重（收益能力 30%，风险控制 25%，风险调整收益 25%，稳定性 10%，相对排名 10%）。总分保留一位小数，并映射为五星至一星评级。
"""


def build_html_report(rows: list[dict[str, Any]]) -> str:
    summary = portfolio_summary(rows)
    sorted_rows = sorted(rows, key=lambda item: item["综合评分(0-10)"], reverse=True)
    bar_html = bar_chart(rows).to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    scatter_html = scatter_chart(rows).to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    radar_html = selected_funds_radar_chart(rows).to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})

    table_rows = "\n".join(
        f"""
        <tr class="{'mock-row' if row.get("数据来源") == "模拟数据" else ''}">
          <td>{row["基金名称"]}</td><td>{str(row["代码"]).zfill(6)}</td><td>{row["基金类型"]}</td>
          <td>{row.get("比较基准", "沪深300")}</td>
          <td>{fmt_pct(to_float(row["近1月收益率"]))}</td><td>{fmt_pct(to_float(row["近1年收益率"]))}</td>
          <td>{fmt_pct(to_float(row.get("近1年超额收益率")))}</td>
          <td>{fmt_pct(to_float(row["近1年最大回撤"]))}</td><td>{row["综合评分(0-10)"]}</td>
          <td>{row["晨星评级"]}</td><td>{row["数据来源"]}</td><td>{row["AI建议"]}</td>
        </tr>
        """
        for row in sorted_rows
    )
    advice_blocks = "\n".join(
        f"""
        <section class="card">
          <div class="card-head">
            <div><h3>{row["基金名称"]}</h3><p>{str(row["代码"]).zfill(6)} · {row["基金类型"]} · {row["数据来源"]}</p></div>
            <strong>{row["综合评分(0-10)"]} / 10 · {row["晨星评级"]}</strong>
          </div>
          <div class="dims">
            <span>收益 {row["收益能力得分"]}</span><span>风险 {row["风险控制得分"]}</span>
            <span>风险调整 {row["风险调整收益得分"]}</span><span>稳定 {row["稳定性得分"]}</span>
            <span>排名 {row["相对排名得分"]}</span>
          </div>
          <pre>{row["详细AI建议"]}</pre>
        </section>
        """
        for row in sorted_rows
    )
    mock_warning = ""
    if any(row.get("数据来源") == "模拟数据" for row in sorted_rows):
        names = "、".join(row["基金名称"] for row in sorted_rows if row.get("数据来源") == "模拟数据")
        mock_warning = f'<div class="warning">注意：以下基金使用模拟数据，仅用于流程验证：{names}</div>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>基金组合可视化报告</title>
  <style>
    body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#0f172a; color:#e2e8f0; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 48px; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    .muted {{ color:#94a3b8; }}
    .disclaimer, .warning {{ border:1px solid rgba(248,113,113,.45); background:#450a0a; color:#fee2e2; border-radius:8px; padding:12px; margin:14px 0; }}
    .grid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; margin:22px 0; }}
    .metric, .card, .chart, .table-wrap {{ border:1px solid rgba(148,163,184,.24); border-radius:10px; background:#111827; padding:16px; }}
    .metric p {{ margin:0; color:#94a3b8; font-size:13px; }} .metric b {{ display:block; margin-top:8px; font-size:20px; color:#f8fafc; }}
    table {{ width:100%; border-collapse: collapse; font-size:13px; }} th, td {{ border-bottom:1px solid #334155; padding:10px; text-align:left; vertical-align:top; }}
    th {{ color:#cbd5e1; }} .chart {{ margin:16px 0; background:white; color:#0f172a; }}
    .mock-row td {{ background:#7f1d1d; color:#fee2e2; }}
    .card {{ margin:14px 0; }} .card-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }}
    .card h3 {{ margin:0; }} .card p {{ margin:4px 0 0; color:#94a3b8; }} .dims {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
    .dims span {{ background:#1f2937; border-radius:6px; padding:6px 8px; font-size:12px; }}
    pre {{ white-space:pre-wrap; font-family:inherit; line-height:1.7; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
    @media (max-width: 560px) {{ .grid {{ grid-template-columns: 1fr; }} .card-head {{ flex-direction:column; }} }}
  </style>
</head>
<body>
<main>
  <h1>基金组合可视化报告</h1>
  <div class="muted">生成时间：{summary["generated_at"]}</div>
  <div class="disclaimer">{COMPLIANCE_DISCLOSURE}</div>
  {mock_warning}
  <div class="grid">
    <div class="metric"><p>总持仓金额</p><b>{summary["total_amount"]:.2f}</b></div>
    <div class="metric"><p>总累计收益</p><b>{summary["total_profit"]:.2f}</b></div>
    <div class="metric"><p>最高评分基金</p><b>{summary["best"]["基金名称"]}</b><span>{summary["best"]["综合评分(0-10)"]} 分</span></div>
    <div class="metric"><p>最低风险基金</p><b>{summary["lowest_risk"]["基金名称"]}</b><span>风险控制 {summary["lowest_risk"]["风险控制得分"]}</span></div>
  </div>
  <section class="table-wrap"><h2>数据表格</h2><table><thead><tr><th>基金</th><th>代码</th><th>类型</th><th>基准</th><th>近1月</th><th>近1年</th><th>超额</th><th>回撤</th><th>评分</th><th>评级</th><th>来源</th><th>建议</th></tr></thead><tbody>{table_rows}</tbody></table></section>
  <section class="chart">{bar_html}</section>
  <section class="chart">{scatter_html}</section>
  <section class="chart">{radar_html}</section>
  <p class="muted">说明：指标在当前基金池内转换为 0-10 百分位分；收益能力优先使用相对比较基准的超额收益，权重按历史样本 IC/IR 估计，样本不足时回退默认权重。历史回测不代表未来表现。</p>
  <h2>详细 AI 建议</h2>
  {advice_blocks}
</main>
</body>
</html>"""


def render_metrics(rows: list[dict[str, Any]]) -> None:
    summary = portfolio_summary(rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总持仓金额", f"{summary['total_amount']:.2f}")
    c2.metric("总累计收益", f"{summary['total_profit']:.2f}")
    c3.metric("最高评分基金", summary["best"]["基金名称"], f"{summary['best']['综合评分(0-10)']} 分")
    c4.metric("最低风险基金", summary["lowest_risk"]["基金名称"], f"风险控制 {summary['lowest_risk']['风险控制得分']}")


def require_mock_data_confirmation(rows: list[dict[str, Any]]) -> bool:
    mock_rows = [row for row in rows if row.get("数据来源") == "模拟数据"]
    if not mock_rows:
        return True

    codes = ",".join(sorted(str(row["代码"]).zfill(6) for row in mock_rows))
    confirm_key = f"mock_data_confirmed_{codes}"
    st.error(
        "检测到以下基金使用模拟数据："
        + "、".join(f"{row['基金名称']}（{str(row['代码']).zfill(6)}）" for row in mock_rows)
        + "。模拟数据只能用于界面流程验证，不能用于判断基金表现。"
    )
    confirmed = st.checkbox("我已了解上述基金使用模拟数据，仍继续查看结果", key=confirm_key)
    if not confirmed:
        st.stop()
        return False
    return True


def render_holding_snapshot_settings() -> tuple[str, bool]:
    with st.sidebar.expander("持仓快照自动更新", expanded=True):
        auto_update = st.checkbox(
            "按最新净值自动更新持仓金额",
            value=bool(st.session_state.get("auto_update_holdings", True)),
            help="用快照日净值和最新净值比例，自动推算当前持有金额、持有收益率和累计收益。",
        )
        snapshot_value = st.session_state.get("holdings_snapshot_date", default_snapshot_date())
        snapshot_date = st.date_input(
            "持仓数据日期",
            value=pd.to_datetime(snapshot_value).date(),
            help="填写你录入这些持有金额和收益率时对应的账户日期，例如 2026-05-27 或 2026-05-28。",
        )
        snapshot_date_text = snapshot_date.strftime("%Y-%m-%d")
        if (
            auto_update != st.session_state.get("auto_update_holdings")
            or snapshot_date_text != st.session_state.get("holdings_snapshot_date")
        ):
            st.session_state.auto_update_holdings = auto_update
            st.session_state.holdings_snapshot_date = snapshot_date_text
            st.cache_data.clear()
            st.session_state.refresh_token += 1
            st.rerun()
        st.caption("自动更新只估算净值涨跌带来的持仓变化；新增申购、赎回、手续费和现金分红仍需手动调整持仓。")
    return snapshot_date_text, auto_update


def main() -> None:
    inject_css()
    render_compliance_footer()
    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0
    if "dashscope_api_key" not in st.session_state:
        st.session_state.dashscope_api_key = ""
    if "holdings" not in st.session_state:
        st.session_state.holdings = default_holdings()
    else:
        st.session_state.holdings = normalize_holdings(st.session_state.holdings)
    if "holdings_snapshot_date" not in st.session_state:
        st.session_state.holdings_snapshot_date = default_snapshot_date()
    if "auto_update_holdings" not in st.session_state:
        st.session_state.auto_update_holdings = True

    st.sidebar.title("AI 智选基金助手")
    with st.sidebar.expander("设置", expanded=False):
        api_key_input = st.text_input(
            "DASHSCOPE_API_KEY",
            value=st.session_state.dashscope_api_key,
            type="password",
            help="仅临时保存在当前 Streamlit 会话中，不写入文件。",
        )
        c1, c2 = st.columns(2)
        if c1.button("应用 Key", use_container_width=True):
            st.session_state.dashscope_api_key = api_key_input.strip()
            st.cache_data.clear()
            st.session_state.refresh_token += 1
            st.rerun()
        if c2.button("清除 Key", use_container_width=True):
            st.session_state.dashscope_api_key = ""
            st.cache_data.clear()
            st.session_state.refresh_token += 1
            st.rerun()
        if st.session_state.dashscope_api_key:
            st.caption("已使用会话中的 DASHSCOPE_API_KEY。")
        elif get_dashscope_api_key():
            st.caption("当前使用已配置的 DASHSCOPE_API_KEY。")
        else:
            st.caption("未配置 Key，将使用本地 mock 建议。")

    render_holdings_editor()
    snapshot_date, auto_update_holdings = render_holding_snapshot_settings()

    score_range = st.sidebar.slider("综合评分范围", 0.0, 10.0, (0.0, 10.0), 0.1)
    fund_type = st.sidebar.selectbox("基金类型", list(TYPE_KEYWORDS.keys()))

    holdings_json = holdings_to_json(st.session_state.holdings)
    with st.spinner("数据加载中，请稍候..."):
        base_rows = load_data(st.session_state.refresh_token, holdings_json, snapshot_date, auto_update_holdings)
    render_manual_override_panel(base_rows)
    rows = apply_manual_overrides(base_rows)
    require_mock_data_confirmation(rows)
    filtered_rows = filter_rows(rows, score_range, fund_type)

    st.sidebar.divider()
    if filtered_rows:
        st.sidebar.download_button(
            "导出 Markdown 报告",
            data=build_markdown_report(filtered_rows).encode("utf-8-sig"),
            file_name="fund_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.sidebar.download_button(
            "导出 HTML 可视化报告",
            data=build_html_report(filtered_rows).encode("utf-8"),
            file_name="fund_report_可视化.html",
            mime="text/html",
            use_container_width=True,
        )
    else:
        st.sidebar.info("当前筛选无数据，无法导出。")

    if st.sidebar.button("刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.session_state.refresh_token += 1
        st.rerun()

    st.title("AI 智选基金助手")
    st.caption("统一应用：数据获取、多维评分、通义千问建议、Markdown/HTML 导出都在本文件内完成。")
    st.info(
        "评分说明：各指标在当前基金池内转换为 0-10 百分位分；收益能力优先看相对比较基准的超额收益；"
        "权重按历史样本 IC/IR 估计，样本不足时回退默认权重。"
    )
    if auto_update_holdings:
        updated_count = sum(1 for row in rows if row.get("持仓已自动更新"))
        st.success(f"持仓已按 {snapshot_date} 快照净值自动更新到最新净值日期，共更新 {updated_count} 只基金。")
    else:
        st.caption("当前未开启持仓自动更新，表格中的持有金额沿用录入快照。")

    if not filtered_rows:
        st.warning("当前筛选条件下没有基金。请调整评分范围或基金类型。")
        return

    render_metrics(filtered_rows)

    st.subheader("基金数据表格")
    table_df = build_table_df(filtered_rows)
    table_event = st.dataframe(
        table_df.style.apply(highlight_mock_rows, axis=1),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "当前持有收益率": st.column_config.NumberColumn("持有收益率(%)", format="%.2f"),
            "快照持有金额": st.column_config.NumberColumn("快照金额", format="%.2f"),
            "持仓快照日期": st.column_config.TextColumn("快照日期"),
            "近1月收益率": st.column_config.NumberColumn("近1月(%)", format="%.2f"),
            "近1年收益率": st.column_config.NumberColumn("近1年(%)", format="%.2f"),
            "近1年超额收益率": st.column_config.NumberColumn("超额收益(%)", format="%.2f"),
            "数据来源": st.column_config.TextColumn("数据来源"),
            "综合评分(0-10)": st.column_config.NumberColumn("综合评分", format="%.1f"),
            "AI建议": st.column_config.TextColumn("简短 AI 建议", width="large"),
        },
    )
    selected_row = selected_row_from_table(table_event, filtered_rows) or max(filtered_rows, key=lambda item: item["综合评分(0-10)"])

    left, right = st.columns(2)
    with left:
        st.plotly_chart(bar_chart(filtered_rows), use_container_width=True)
    with right:
        st.plotly_chart(scatter_chart(filtered_rows), use_container_width=True)

    st.subheader("所选基金五维雷达图")
    st.caption(f"当前选择：{selected_row['基金名称']}（{str(selected_row['代码']).zfill(6)}）")
    st.plotly_chart(radar_chart(selected_row), use_container_width=True)

    render_backtest_simulator(filtered_rows, selected_row)

    st.subheader("详细 AI 建议")
    for row in filtered_rows:
        title = f"{row['基金名称']}｜{str(row['代码']).zfill(6)}｜{row['综合评分(0-10)']} 分｜{row['晨星评级']}"
        with st.expander(title, expanded=row["代码"] == selected_row["代码"]):
            st.markdown(
                f"""
                <div class="hint-card">
                  <strong>{row["基金名称"]}</strong>
                  <div class="small-muted">类型：{row["基金类型"]} ｜ 基准：{row.get("比较基准", "沪深300")} ｜ 数据来源：{row["数据来源"]} ｜ 最新净值日期：{row["最新净值日期"]}</div>
                  <div class="small-muted">{row.get("持仓更新说明", "")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(row["详细AI建议"])


if __name__ == "__main__":
    main()

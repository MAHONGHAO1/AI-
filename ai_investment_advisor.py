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


REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "requests": "requests",
    "plotly": "plotly",
    "openai": "openai",
    "tabulate": "tabulate",
}


def ensure_packages() -> None:
    missing = [pip_name for import_name, pip_name in REQUIRED_PACKAGES.items() if importlib.util.find_spec(import_name) is None]
    if not missing:
        return

    base_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--prefer-binary",
        "--timeout",
        "120",
        "--retries",
        "5",
        "--progress-bar",
        "off",
        *missing,
    ]
    mirror_cmd = [
        *base_cmd,
        "-i",
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "--trusted-host",
        "pypi.tuna.tsinghua.edu.cn",
    ]

    print(f"检测到缺失依赖：{missing}，开始安装...", flush=True)
    for label, cmd in [("默认 PyPI", base_cmd), ("清华 PyPI 镜像", mirror_cmd)]:
        try:
            print(f"正在通过{label}安装依赖...", flush=True)
            subprocess.check_call(cmd)
            return
        except subprocess.CalledProcessError as exc:
            print(f"{label}安装失败：{exc}", flush=True)

    raise RuntimeError(
        "依赖自动安装失败。请在 PowerShell 手动运行：\n"
        "python -m pip install streamlit pandas numpy requests plotly openai lxml tabulate "
        "-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --timeout 120"
    )


ensure_packages()


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
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    df = ak.fund_open_fund_info_em(fund=code, indicator="单位净值走势")
    if df is None or df.empty:
        raise ValueError("AkShare 返回空数据")
    date_col = next((c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"), None)
    nav_col = next((c for c in df.columns if "单位净值" in str(c) or "净值" in str(c)), None)
    if date_col is None or nav_col is None:
        raise ValueError(f"AkShare 字段不符合预期：{list(df.columns)}")
    out = pd.DataFrame({"date": df[date_col].map(safe_date), "nav": df[nav_col].map(to_float)})
    out = out.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("AkShare 清洗后无有效净值")
    return out


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
    out = pd.DataFrame({"date": df[date_col].map(safe_date), "nav": df[nav_col].map(to_float)})
    out = out.dropna(subset=["date", "nav"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("东方财富清洗后无有效净值")
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
    for source, fetcher in [("AkShare", fetch_nav_with_akshare), ("东方财富", fetch_nav_with_eastmoney)]:
        try:
            return fetcher(code), source, ""
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    return make_mock_nav(code), "模拟数据", "；".join(errors)


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


def score_higher_better(value: float, zero_at: float, ten_at: float, default: float = 5.0) -> float:
    if math.isnan(value) or ten_at == zero_at:
        return default
    return clamp((value - zero_at) / (ten_at - zero_at) * 10)


def score_lower_better(value: float, ten_at: float, zero_at: float, default: float = 5.0) -> float:
    if math.isnan(value) or zero_at == ten_at:
        return default
    return clamp((zero_at - value) / (zero_at - ten_at) * 10)


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


def calculate_composite_score(fund_data: dict) -> dict:
    ret_1m = to_float(fund_data.get("近1月收益率"))
    ret_3m = to_float(fund_data.get("近3月收益率"))
    ret_1y = to_float(fund_data.get("近1年收益率"))
    mdd = to_float(fund_data.get("近1年最大回撤"))
    sharpe = to_float(fund_data.get("近1年夏普比率"))
    relative_score = to_float(fund_data.get("相对排名分", 5.0))

    ret_1y_score = score_higher_better(ret_1y, zero_at=-20.0, ten_at=20.0)
    ret_1m_score = score_higher_better(ret_1m, zero_at=-10.0, ten_at=10.0)
    return_score = ret_1y_score * 0.7 + ret_1m_score * 0.3
    risk_score = score_higher_better(mdd, zero_at=-30.0, ten_at=-5.0)
    sharpe_score = score_higher_better(sharpe, zero_at=1.0, ten_at=2.0, default=0.0)

    if math.isnan(ret_1y) or math.isnan(mdd) or abs(mdd) <= 0:
        calmar_score = 0.0
    else:
        calmar_score = score_higher_better(ret_1y / abs(mdd), zero_at=0.0, ten_at=5.0, default=0.0)
    risk_adjusted_score = (sharpe_score + calmar_score) / 2

    if math.isnan(ret_3m) or math.isnan(ret_1y):
        stability_score = 5.0
    else:
        stability_score = score_lower_better(abs(ret_3m - ret_1y), ten_at=0.0, zero_at=30.0)

    dimension_scores = {
        "收益能力": round(return_score, 1),
        "风险控制": round(risk_score, 1),
        "风险调整收益": round(risk_adjusted_score, 1),
        "稳定性": round(stability_score, 1),
        "相对排名": round(clamp(relative_score), 1),
    }
    total_score = round(
        dimension_scores["收益能力"] * 0.30
        + dimension_scores["风险控制"] * 0.25
        + dimension_scores["风险调整收益"] * 0.25
        + dimension_scores["稳定性"] * 0.10
        + dimension_scores["相对排名"] * 0.10,
        1,
    )
    return {"total_score": total_score, "dimension_scores": dimension_scores, "grade": grade_from_score(total_score)}


def relative_rank_score(position: int, total: int) -> float:
    if total <= 1:
        return 10.0
    percentile = position / (total - 1)
    if percentile <= 0.2:
        return 10.0
    if percentile >= 0.8:
        return 2.0
    return round(10.0 - ((percentile - 0.2) / 0.6) * 8.0, 1)


def apply_composite_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["相对排名分"] = 5.0
        row["_预评分"] = calculate_composite_score(row)["total_score"]

    ranked_rows = sorted(rows, key=lambda item: item["_预评分"], reverse=True)
    for position, row in enumerate(ranked_rows):
        row["相对排名分"] = relative_rank_score(position, len(ranked_rows))
        result = calculate_composite_score(row)
        row["综合评分(0-10)"] = result["total_score"]
        row["晨星评级"] = result["grade"]
        row["维度分数"] = result["dimension_scores"]
        for dimension, score in result["dimension_scores"].items():
            row[f"{dimension}得分"] = score
        row.pop("_预评分", None)


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


def parse_qwen_advice_pair(text: str) -> tuple[str, str]:
    """解析一次 AI 调用返回的简短和详细建议。"""
    short_match = re.search(r"【简短建议】\s*(.*?)(?=【详细建议】|$)", text, re.S)
    detail_match = re.search(r"【详细建议】\s*(.*)$", text, re.S)
    short_text = short_match.group(1).strip() if short_match else ""
    detail_text = detail_match.group(1).strip() if detail_match else ""
    return short_text, detail_text


def call_qwen_advice_pair(row: dict[str, Any]) -> tuple[str, str]:
    """一次通义千问调用同时生成简短建议和详细建议，减少 API 调用次数。"""
    api_key = st.session_state.get("dashscope_api_key") or os.getenv("DASHSCOPE_API_KEY")
    mock_short = mock_short_advice(
        row["基金名称"],
        to_float(row["近1年收益率"]),
        to_float(row["近1年最大回撤"]),
        to_float(row["当前持有收益率"]),
    )
    mock_detail = mock_detailed_advice(row)
    if not api_key:
        return mock_short, mock_detail

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
近1年最大回撤：{fmt_pct(to_float(row["近1年最大回撤"]))}
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
            return short_text or mock_short, detail_text or mock_detail
    except Exception as exc:
        show_notice_once("qwen_call_failed", f"通义千问调用失败，已使用本地 mock 建议：{exc}", "warning")

    return mock_short, mock_detail


def analyze_fund(fund: dict[str, str], holdings: dict[str, dict[str, float]]) -> dict[str, Any]:
    code = fund["code"]
    name = fund["name"]
    nav_df, source, error_msg = fetch_nav_history(code)
    latest = nav_df.iloc[-1]
    ret_1m = calc_period_return(nav_df, 30)
    ret_3m = calc_period_return(nav_df, 90)
    ret_1y = calc_period_return(nav_df, 365)
    mdd = calc_max_drawdown(nav_df)
    sharpe = calc_sharpe(nav_df)
    holding = holdings.get(code, {"amount": 0.0, "hold_ret": 0.0, "cum_profit": 0.0})
    return {
        "基金名称": name,
        "代码": code,
        "基金类型": classify_fund(name),
        "最新净值日期": latest["date"].strftime("%Y-%m-%d"),
        "单位净值": float(latest["nav"]),
        "近1月收益率": ret_1m,
        "近3月收益率": ret_3m,
        "近1年收益率": ret_1y,
        "近1年最大回撤": mdd,
        "近1年夏普比率": sharpe,
        "当前持有金额": float(holding.get("amount", 0.0)),
        "当前持有收益率": float(holding.get("hold_ret", 0.0)),
        "累计收益": float(holding.get("cum_profit", 0.0)),
        "数据来源": source,
        "错误信息": error_msg,
    }


def analyze_fund_with_mock(
    fund: dict[str, str],
    nav_df: pd.DataFrame,
    error_msg: str,
    holdings: dict[str, dict[str, float]],
) -> dict[str, Any]:
    code = fund["code"]
    name = fund["name"]
    latest = nav_df.iloc[-1]
    ret_1m = calc_period_return(nav_df, 30)
    ret_3m = calc_period_return(nav_df, 90)
    ret_1y = calc_period_return(nav_df, 365)
    mdd = calc_max_drawdown(nav_df)
    sharpe = calc_sharpe(nav_df)
    holding = holdings.get(code, {"amount": 0.0, "hold_ret": 0.0, "cum_profit": 0.0})
    return {
        "基金名称": name,
        "代码": code,
        "基金类型": classify_fund(name),
        "最新净值日期": latest["date"].strftime("%Y-%m-%d"),
        "单位净值": float(latest["nav"]),
        "近1月收益率": ret_1m,
        "近3月收益率": ret_3m,
        "近1年收益率": ret_1y,
        "近1年最大回撤": mdd,
        "近1年夏普比率": sharpe,
        "当前持有金额": float(holding.get("amount", 0.0)),
        "当前持有收益率": float(holding.get("hold_ret", 0.0)),
        "累计收益": float(holding.get("cum_profit", 0.0)),
        "数据来源": "模拟数据",
        "错误信息": error_msg,
    }


@st.cache_data(ttl=86400, show_spinner="数据加载中，请稍候...")
def load_data(refresh_token: int = 0, holdings_json: str = "") -> list[dict[str, Any]]:
    _ = refresh_token
    try:
        holdings_list = normalize_holdings(json.loads(holdings_json or "[]"))
    except Exception:
        holdings_list = default_holdings()
    holdings = holdings_to_map(holdings_list)
    rows = []
    for holding_item in holdings_list:
        fund = {"code": holding_item["code"], "name": holding_item["name"]}
        try:
            rows.append(analyze_fund(fund, holdings))
        except Exception as exc:
            rows.append(analyze_fund_with_mock(fund, make_mock_nav(fund["code"]), str(exc), holdings))

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
            row["详细AI建议"] = mock_detailed_advice(row)
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
            "近1月收益率",
            "近1年收益率",
            "综合评分(0-10)",
            "晨星评级",
            "AI建议",
        ]
    ].copy()
    table_df["代码"] = table_df["代码"].astype(str).str.zfill(6)
    for col in ["当前持有金额", "当前持有收益率", "近1月收益率", "近1年收益率", "综合评分(0-10)"]:
        table_df[col] = table_df[col].map(lambda x: round(to_float(x), 2))
    return table_df


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


def simulate_monthly_investment(nav_df: pd.DataFrame, months: int, monthly_amount: float) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """模拟过去 N 个月每月定投固定金额，并按最新净值估算当前收益。"""
    if nav_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    df = nav_df.copy().sort_values("date").reset_index(drop=True)
    latest_date = df.iloc[-1]["date"]
    latest_nav = float(df.iloc[-1]["nav"])
    start_date = latest_date - pd.DateOffset(months=months)
    recent = df[df["date"] >= start_date].copy()
    if recent.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    recent["month"] = recent["date"].dt.to_period("M")
    purchases = recent.groupby("month", as_index=False).first().tail(months).copy()
    purchases["投入金额"] = float(monthly_amount)
    purchases["买入份额"] = purchases["投入金额"] / purchases["nav"]
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
        market_values.append(shares * float(day["nav"]))
    curve["累计投入"] = invested_values
    curve["组合当前价值"] = market_values
    curve["收益率"] = np.where(curve["累计投入"] > 0, curve["组合当前价值"] / curve["累计投入"] - 1, 0.0)

    total_invested = float(purchases["投入金额"].sum())
    total_shares = float(purchases["买入份额"].sum())
    current_value = total_shares * latest_nav
    profit = current_value - total_invested
    profit_rate = profit / total_invested * 100 if total_invested > 0 else 0.0
    summary = {
        "total_invested": total_invested,
        "current_value": current_value,
        "profit": profit,
        "profit_rate": profit_rate,
        "latest_nav": latest_nav,
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

    fund = options[fund_label]
    try:
        nav_df, source, err = load_nav_for_backtest(str(fund["代码"]).zfill(6), st.session_state.refresh_token)
        summary, purchases, curve = simulate_monthly_investment(nav_df, int(months), float(monthly_amount))
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
    st.caption(f"回测数据来源：{source}。假设每月第一个可用净值日买入，不考虑申购费、赎回费、分红和税费。")
    if err:
        st.caption(f"数据提示：{err}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["累计投入"], mode="lines", name="累计投入"))
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["组合当前价值"], mode="lines", name="组合价值"))
    fig.update_layout(height=380, title="定投组合价值曲线", margin=dict(l=30, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

    purchase_table = purchases[["date", "nav", "投入金额", "买入份额", "当前价值", "当前盈亏"]].copy()
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
        ["基金名称", "代码", "近1月收益率", "近1年收益率", "近1年最大回撤", "综合评分(0-10)", "晨星评级", "AI建议"]
    ].copy()
    for col in ["近1月收益率", "近1年收益率", "近1年最大回撤"]:
        table_df[col] = table_df[col].map(lambda x: fmt_pct(to_float(x)))

    detail_df = df[
        [
            "基金名称",
            "代码",
            "最新净值日期",
            "单位净值",
            "近3月收益率",
            "近1年夏普比率",
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
    detail_df["近3月收益率"] = detail_df["近3月收益率"].map(lambda x: fmt_pct(to_float(x)))
    detail_df["近1年夏普比率"] = detail_df["近1年夏普比率"].map(lambda x: fmt_num(to_float(x), 2))
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

> 免责声明：本报告由公开净值数据、简化量化规则和 AI 文本生成，不能替代专业投顾意见；脚本不执行任何交易操作。
{warning_text}
## 核心评分与建议

{table_df.to_markdown(index=False)}

## 持仓与数据明细

{detail_df.to_markdown(index=False)}

综合评分最高的 3 只基金：{top_text}。

综合评分最低的 3 只基金：{bottom_text}。

多维度评分说明：收益能力按近 1 年收益率和近 1 月收益率综合计算，其中近 1 年收益率按 -20% 到 20% 映射到 0-10 分，超过 20% 按 10 分封顶；风险控制按近 1 年最大回撤评分，风险调整收益由夏普比率和卡玛比率共同决定，稳定性衡量近 3 月与近 1 年收益率差异，相对排名依据本次基金池内综合预评分百分位给分。

权重说明：收益能力 30%，风险控制 25%，风险调整收益 25%，稳定性 10%，相对排名 10%；总分保留一位小数，并映射为五星至一星评级。
"""


def build_html_report(rows: list[dict[str, Any]]) -> str:
    summary = portfolio_summary(rows)
    sorted_rows = sorted(rows, key=lambda item: item["综合评分(0-10)"], reverse=True)
    bar_html = bar_chart(rows).to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    scatter_html = scatter_chart(rows).to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    radar_html = selected_funds_radar_chart(rows).to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})

    table_rows = "\n".join(
        f"""
        <tr>
          <td>{row["基金名称"]}</td><td>{str(row["代码"]).zfill(6)}</td><td>{row["基金类型"]}</td>
          <td>{fmt_pct(to_float(row["近1月收益率"]))}</td><td>{fmt_pct(to_float(row["近1年收益率"]))}</td>
          <td>{fmt_pct(to_float(row["近1年最大回撤"]))}</td><td>{row["综合评分(0-10)"]}</td>
          <td>{row["晨星评级"]}</td><td>{row["AI建议"]}</td>
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
    .grid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; margin:22px 0; }}
    .metric, .card, .chart, .table-wrap {{ border:1px solid rgba(148,163,184,.24); border-radius:10px; background:#111827; padding:16px; }}
    .metric p {{ margin:0; color:#94a3b8; font-size:13px; }} .metric b {{ display:block; margin-top:8px; font-size:20px; color:#f8fafc; }}
    table {{ width:100%; border-collapse: collapse; font-size:13px; }} th, td {{ border-bottom:1px solid #334155; padding:10px; text-align:left; vertical-align:top; }}
    th {{ color:#cbd5e1; }} .chart {{ margin:16px 0; background:white; color:#0f172a; }}
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
  <div class="grid">
    <div class="metric"><p>总持仓金额</p><b>{summary["total_amount"]:.2f}</b></div>
    <div class="metric"><p>总累计收益</p><b>{summary["total_profit"]:.2f}</b></div>
    <div class="metric"><p>最高评分基金</p><b>{summary["best"]["基金名称"]}</b><span>{summary["best"]["综合评分(0-10)"]} 分</span></div>
    <div class="metric"><p>最低风险基金</p><b>{summary["lowest_risk"]["基金名称"]}</b><span>风险控制 {summary["lowest_risk"]["风险控制得分"]}</span></div>
  </div>
  <section class="table-wrap"><h2>数据表格</h2><table><thead><tr><th>基金</th><th>代码</th><th>类型</th><th>近1月</th><th>近1年</th><th>回撤</th><th>评分</th><th>评级</th><th>建议</th></tr></thead><tbody>{table_rows}</tbody></table></section>
  <section class="chart">{bar_html}</section>
  <section class="chart">{scatter_html}</section>
  <section class="chart">{radar_html}</section>
  <p class="muted">说明：收益能力中的近 1 年收益率按 -20% 到 20% 映射评分，超过 20% 按 10 分封顶；雷达图仅展示综合评分前 3 名和后 3 名，避免曲线过度叠加。</p>
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


def main() -> None:
    inject_css()
    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0
    if "dashscope_api_key" not in st.session_state:
        st.session_state.dashscope_api_key = ""
    if "holdings" not in st.session_state:
        st.session_state.holdings = default_holdings()
    else:
        st.session_state.holdings = normalize_holdings(st.session_state.holdings)

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
        elif os.getenv("DASHSCOPE_API_KEY"):
            st.caption("当前使用系统环境变量 DASHSCOPE_API_KEY。")
        else:
            st.caption("未配置 Key，将使用本地 mock 建议。")

    render_holdings_editor()

    score_range = st.sidebar.slider("综合评分范围", 1.0, 10.0, (1.0, 10.0), 0.1)
    fund_type = st.sidebar.selectbox("基金类型", list(TYPE_KEYWORDS.keys()))

    holdings_json = holdings_to_json(st.session_state.holdings)
    with st.spinner("数据加载中，请稍候..."):
        base_rows = load_data(st.session_state.refresh_token, holdings_json)
    render_manual_override_panel(base_rows)
    rows = apply_manual_overrides(base_rows)
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
    st.info("评分说明：收益能力中的近 1 年收益率按 -20% 到 20% 映射到 0-10 分，超过 20% 按 10 分封顶。")

    if not filtered_rows:
        st.warning("当前筛选条件下没有基金。请调整评分范围或基金类型。")
        return

    render_metrics(filtered_rows)

    st.subheader("基金数据表格")
    table_df = build_table_df(filtered_rows)
    table_event = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "当前持有收益率": st.column_config.NumberColumn("持有收益率(%)", format="%.2f"),
            "近1月收益率": st.column_config.NumberColumn("近1月(%)", format="%.2f"),
            "近1年收益率": st.column_config.NumberColumn("近1年(%)", format="%.2f"),
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
                  <div class="small-muted">类型：{row["基金类型"]} ｜ 数据来源：{row["数据来源"]} ｜ 最新净值日期：{row["最新净值日期"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(row["详细AI建议"])


if __name__ == "__main__":
    main()

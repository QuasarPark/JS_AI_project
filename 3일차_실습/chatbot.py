"""OpenAI API + Streamlit 간단 챗봇 (3일차 실습)."""

import json
import os
from datetime import datetime
from pathlib import Path

import pytz
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
MODEL = "gpt-5.4-mini"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when needed. "
    "Answer in Korean unless the user asks otherwise."
)

CITY_TZ = {
    "서울": "Asia/Seoul",
    "seoul": "Asia/Seoul",
    "뉴욕": "America/New_York",
    "new york": "America/New_York",
    "도쿄": "Asia/Tokyo",
    "tokyo": "Asia/Tokyo",
    "런던": "Europe/London",
    "london": "Europe/London",
}


def get_city_time_tz(city: str) -> str:
    key = city.strip().lower()
    tz_name = CITY_TZ.get(key)
    if not tz_name:
        return json.dumps({"error": f"지원하지 않는 도시: {city}"}, ensure_ascii=False)

    now = datetime.now(pytz.timezone(tz_name)).strftime("%Y-%m-%d %H:%M:%S")
    return json.dumps(
        {"city": city, "timezone": tz_name, "current_time": now},
        ensure_ascii=False,
    )


def get_us_stock_price(ticker: str) -> str:
    symbol = ticker.strip().upper()
    try:
        hist = yf.Ticker(symbol).history(period="2d")
        if hist.empty:
            return json.dumps({"error": f"{symbol} 데이터가 없습니다."}, ensure_ascii=False)

        latest = hist.iloc[-1]
        prev_close = float(latest["Close"]) if "Close" in latest else None
        open_price = float(latest["Open"]) if "Open" in latest else None

        return json.dumps(
            {
                "ticker": symbol,
                "open": round(open_price, 2) if open_price is not None else None,
                "close": round(prev_close, 2) if prev_close is not None else None,
                "currency": "USD",
                "source": "yfinance",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


WMO_WEATHER = {
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "서리 안개",
    51: "가벼운 이슬비",
    53: "이슬비",
    55: "강한 이슬비",
    61: "약한 비",
    63: "비",
    65: "강한 비",
    71: "약한 눈",
    73: "눈",
    75: "강한 눈",
    80: "약한 소나기",
    81: "소나기",
    82: "강한 소나기",
    95: "뇌우",
    96: "우박을 동반한 뇌우",
    99: "강한 우박을 동반한 뇌우",
}


def get_location_weather(location: str) -> str:
    """Open-Meteo API로 지역의 현재 날씨를 조회합니다."""
    name = location.strip()
    if not name:
        return json.dumps({"error": "지역 이름을 입력해주세요."}, ensure_ascii=False)

    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "ko", "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results") or []
        if not results:
            return json.dumps(
                {"error": f"지역을 찾을 수 없습니다: {location}"},
                ensure_ascii=False,
            )

        place = results[0]
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        current = weather_resp.json().get("current", {})
        code = current.get("weather_code")

        return json.dumps(
            {
                "location": place.get("name", name),
                "region": place.get("admin1"),
                "country": place.get("country"),
                "temperature_c": current.get("temperature_2m"),
                "feels_like_c": current.get("apparent_temperature"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "condition": WMO_WEATHER.get(code, f"코드 {code}"),
                "observed_at": current.get("time"),
                "source": "open-meteo",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


TZ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_city_time_tz",
            "description": "도시의 시간대를 반영해 현재 시간을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

STOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_price",
            "description": "미국 주식 티커의 최근 가격(시가/종가)을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    }
]

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_location_weather",
            "description": "지역(도시) 이름으로 현재 날씨(기온, 체감온도, 습도, 풍속, 날씨 상태)를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "도시 또는 지역 이름 (예: 서울, Tokyo, New York)",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

ALL_TOOLS = TZ_TOOLS + STOCK_TOOLS + WEATHER_TOOLS
TOOL_FUNCTIONS = {
    "get_city_time_tz": get_city_time_tz,
    "get_us_stock_price": get_us_stock_price,
    "get_location_weather": get_location_weather,
}


def load_api_key() -> str:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(f"{ENV_PATH}에 OPENAI_API_KEY=sk-... 를 설정하세요.")
        st.stop()
    return api_key


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _assistant_message(msg) -> dict:
    payload: dict = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return payload


def _run_tool_calls(msg, messages: list[dict]) -> None:
    messages.append(_assistant_message(msg))
    for tc in msg.tool_calls:
        fn_name = tc.function.name
        fn_args = json.loads(tc.function.arguments or "{}")
        result = TOOL_FUNCTIONS[fn_name](**fn_args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
        )


def chat(messages: list[dict]):
    client = get_client(st.session_state.api_key)

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=ALL_TOOLS,
            temperature=0.1,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            _run_tool_calls(msg, messages)
            continue

        content = msg.content or ""
        messages.append({"role": "assistant", "content": content})
        yield content
        return


def main() -> None:
    st.set_page_config(page_title="OpenAI 챗봇", page_icon="💬")
    st.title("💬 OpenAI 챗봇")
    st.caption("3일차 실습 — Tool Calling (도시 시간 · 미국 주식 · 날씨)")

    if "api_key" not in st.session_state:
        st.session_state.api_key = load_api_key()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    with st.sidebar:
        st.markdown(f"**모델:** `{MODEL}`")
        st.markdown("**등록된 Tool**")
        st.markdown("- `get_city_time_tz` — 서울·뉴욕·도쿄·런던 등")
        st.markdown("- `get_us_stock_price` — AAPL, TSLA 등 미국 주식")
        st.markdown("- `get_location_weather` — 도시별 현재 날씨")
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()

    for message in st.session_state.messages:
        if message["role"] in ("system", "tool"):
            continue
        if (
            message["role"] == "assistant"
            and message.get("tool_calls")
            and not message.get("content")
        ):
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.write_stream(chat(st.session_state.messages))


if __name__ == "__main__":
    main()

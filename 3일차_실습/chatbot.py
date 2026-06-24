"""OpenAI API + Streamlit 간단 챗봇 (3일차 실습)."""

from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
MODEL = "gpt-5.4-mini"
SYSTEM_PROMPT = "You are a helpful assistant. Answer in Korean unless the user asks otherwise."


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


def chat(messages: list[dict]) -> str:
    client = get_client(st.session_state.api_key)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        if text:
            chunks.append(text)
            yield text
    st.session_state.messages.append(
        {"role": "assistant", "content": "".join(chunks)}
    )


def main() -> None:
    st.set_page_config(page_title="OpenAI 챗봇", page_icon="💬")
    st.title("💬 OpenAI 챗봇")
    st.caption("3일차 실습 — `.env`의 OPENAI_API_KEY 사용")

    if "api_key" not in st.session_state:
        st.session_state.api_key = load_api_key()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    with st.sidebar:
        st.markdown(f"**모델:** `{MODEL}`")
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()

    for message in st.session_state.messages:
        if message["role"] == "system":
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

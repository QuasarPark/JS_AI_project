import os
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """PyMuPDF로 PDF 전체 텍스트를 추출한다."""
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n------------------------\n"
    return full_text


def pdf_to_text(pdf_path: str | Path) -> Path:
    """PDF를 텍스트로 추출해 같은 디렉터리에 .txt로 저장하고, 저장 경로를 반환한다."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    txt_path = pdf_path.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    return txt_path


def summarize_pdf(pdf_path: str | Path, save: bool = False) -> str:
    """
    PDF 파일 경로를 입력받아 요약을 생성하고, save=True이면 같은 디렉터리에 summary.txt를 저장한 뒤 요약 문자열을 반환한다.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = f"""
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고,

    작성해야 하는 포맷은 다음과 같음
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)

    ## 저자 소개


    ============= 이하 텍스트 ================
    {text[:10000]}

    """

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        temperature=0.1,
        messages=[{"role": "system", "content": system_prompt}],
    )
    summary = response.choices[0].message.content

    if save:
        summary_path = pdf_path.parent / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

    if save:
        return summary, summary_path
    else:
        return summary

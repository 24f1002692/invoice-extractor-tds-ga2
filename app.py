import re
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Local LLM (runs in-process, no external API calls)
# ---------------------------------------------------------------------------
from transformers import pipeline

MODEL_NAME = "google/flan-t5-small"
_llm = None


def get_llm():
    """Lazy-load so the process starts fast; model loads on first request."""
    global _llm
    if _llm is None:
        _llm = pipeline("text2text-generation", model=MODEL_NAME)
    return _llm


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class InvoiceFields(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

    @field_validator("currency")
    @classmethod
    def currency_upper(cls, v):
        return v.upper()[:3] if v else "USD"


app = FastAPI()


# ---------------------------------------------------------------------------
# Regex-based deterministic extraction (primary source of truth for
# amount / currency / date, since these follow strict formats)
# ---------------------------------------------------------------------------
CURRENCY_RE = re.compile(r"\b(USD|EUR|GBP)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
AMOUNT_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
VENDOR_LABEL_RE = re.compile(r"(?:vendor|from|company|bill\s*to|seller)\s*[:\-]?\s*(.+)", re.IGNORECASE)


def extract_currency(text: str) -> Optional[str]:
    m = CURRENCY_RE.search(text)
    return m.group(1).upper() if m else None


def extract_date(text: str) -> Optional[str]:
    m = DATE_RE.search(text)
    return m.group(1) if m else None


def extract_amount(text: str) -> Optional[float]:
    # prefer a number that sits near an amount/total/due keyword
    for line in text.splitlines():
        if re.search(r"amount|total|due|balance", line, re.IGNORECASE):
            m = AMOUNT_RE.search(line.replace(",", ""))
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    pass
    # fallback: any plausible number in the whole text
    candidates = [float(x.replace(",", "")) for x in AMOUNT_RE.findall(text)]
    candidates = [c for c in candidates if 0 < c < 1_000_000]
    return candidates[0] if candidates else None


def extract_vendor_regex(text: str) -> Optional[str]:
    m = VENDOR_LABEL_RE.search(text)
    if m:
        # cut at line end / next label
        return m.group(1).split("\n")[0].strip(" .")
    return None


def extract_vendor_llm(text: str) -> Optional[str]:
    try:
        llm = get_llm()
        prompt = (
            "Extract only the company or vendor name from this invoice text. "
            "Reply with the name only, nothing else.\n\n" + text
        )
        out = llm(prompt, max_new_tokens=20)[0]["generated_text"].strip()
        # guard against empty/garbage LLM output
        out = out.strip(" .\"'")
        return out if out else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/extract")
def extract(payload: dict = Body(...)):
    text = (payload or {}).get("text", "")

    if not isinstance(text, str) or not text.strip():
        return JSONResponse(
            status_code=422,
            content={"error": "text field is required and must be non-empty"},
        )

    try:
        vendor = extract_vendor_regex(text) or extract_vendor_llm(text) or "Unknown Vendor"
        amount = extract_amount(text)
        currency = extract_currency(text) or "USD"
        date = extract_date(text)

        if amount is None:
            amount = 0.0
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        result = InvoiceFields(
            vendor=vendor,
            amount=round(float(amount), 2),
            currency=currency,
            date=date,
        )
        return result.model_dump()

    except Exception:
        # never crash with 500 — best-effort valid JSON
        return InvoiceFields(
            vendor="Unknown Vendor",
            amount=0.0,
            currency="USD",
            date=datetime.utcnow().strftime("%Y-%m-%d"),
        ).model_dump()


@app.get("/")
def health():
    return {"status": "ok"}
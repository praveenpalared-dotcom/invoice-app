import re
import sys
from datetime import datetime
from typing import Optional
import pandas as pd

from ocr_engine import extract_text

_RE_INV_NUM = re.compile(
    r"(?:Invoice\s*#?\s*|INV[-\s]?|No\.?\s*)([A-Z0-9\-]{4,20})",
    re.IGNORECASE,
)

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d-%b-%Y",
]

_RE_DATE = re.compile(
    r"""
    (?:Date[:\s]*)?
    (
        \d{4}[-/]\d{1,2}[-/]\d{1,2}
      | \d{1,2}[-/]\d{1,2}[-/]\d{4}
      | (?:Jan|Feb|Mar|Apr|May|Jun|
           Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*
        [\s,]+\d{1,2}[,\s]+\d{4}
      | \d{1,2}[\s\-]
        (?:Jan|Feb|Mar|Apr|May|Jun|
           Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*
        [\s\-]\d{4}
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_RE_VENDOR = re.compile(
    r"(?:From|Vendor|Supplier|Billed?\s+By)[:\s]+(.+)",
    re.IGNORECASE,
)

_RE_TOTAL = re.compile(
    r"(?:Grand\s+Total|Total\s+Due|Amount\s+Due|Invoice\s+Total)[:\s]*\$?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_RE_LINE_ITEM = re.compile(
    r"^(.+?)\s{2,}(\d+)\s{2,}\$?([\d,]+\.\d{2})\s{2,}\$?([\d,]+\.\d{2})\s*$",
    re.MULTILINE,
)


def _clean_amount(raw: str) -> float:
    return float(raw.replace(",", "").replace("$", "").strip())


def _parse_date(raw: str) -> Optional[str]:
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_invoice_number(text: str) -> Optional[str]:
    match = _RE_INV_NUM.search(text)
    return match.group(1).strip() if match else None


def extract_date(text: str) -> Optional[str]:
    match = _RE_DATE.search(text)
    if not match:
        return None
    return _parse_date(match.group(1))


def extract_vendor(text: str) -> Optional[str]:
    match = _RE_VENDOR.search(text)
    if not match:
        return None
    vendor = match.group(1).split("\n")[0]
    return re.sub(r"[^A-Za-z0-9 .,'&-]", "", vendor).strip()


def extract_grand_total(text: str) -> Optional[float]:
    match = _RE_TOTAL.search(text)
    if not match:
        return None
    try:
        return _clean_amount(match.group(1))
    except ValueError:
        return None


def extract_line_items(text: str) -> list[dict]:
    items = []
    for m in _RE_LINE_ITEM.finditer(text):
        try:
            items.append({
                "description": m.group(1).strip(),
                "qty":         int(m.group(2)),
                "unit_price":  _clean_amount(m.group(3)),
                "line_total":  _clean_amount(m.group(4)),
            })
        except (ValueError, IndexError):
            continue
    return items


def parse_invoice(image_path: str) -> dict:
    text = extract_text(image_path)

    inv_number = extract_invoice_number(text)
    date = extract_date(text)
    vendor = extract_vendor(text)
    grand_total = extract_grand_total(text)
    line_items = extract_line_items(text)

    calc_total = round(sum(i["line_total"] for i in line_items), 2)

    return {
        "file_path":        image_path,
        "invoice_number":   inv_number or "UNKNOWN",
        "invoice_date":     date or "UNKNOWN",
        "vendor_name":      vendor or "UNKNOWN",
        "grand_total":      grand_total,
        "calculated_total": calc_total if line_items else None,
        "num_line_items":   len(line_items),
        "line_items":       line_items,
        "raw_text":         text,
    }


def parse_invoices_to_df(image_paths: list[str]) -> pd.DataFrame:
    records = []
    for path in image_paths:
        print(f"  Parsing: {path}")
        try:
            rec = parse_invoice(path)
        except Exception as exc:
            print(f"    ⚠ Skipped {path}: {exc}")
            rec = {
                "file_path":        path,
                "invoice_number":   "ERROR",
                "invoice_date":     "ERROR",
                "vendor_name":      "ERROR",
                "grand_total":      None,
                "calculated_total": None,
                "num_line_items":   0,
                "line_items":       [],
                "raw_text":         str(exc),
            }
        rec["line_items"] = str(rec["line_items"])
        records.append(rec)

    df = pd.DataFrame(records)
    df["grand_total"]      = pd.to_numeric(df["grand_total"],      errors="coerce")
    df["calculated_total"] = pd.to_numeric(df["calculated_total"], errors="coerce")
    df["num_line_items"]   = pd.to_numeric(df["num_line_items"],   errors="coerce").fillna(0).astype(int)
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <image_path>")
        sys.exit(1)

    result = parse_invoice(sys.argv[1])
    print("\n── Extracted Fields ──────────────────────────────")
    for k, v in result.items():
        if k not in ("raw_text", "line_items"):
            print(f"  {k:<22}: {v}")
    print("\n── Line Items ────────────────────────────────────")
    for item in result["line_items"]:
        print(f"  {item}")
    print("\n── Raw OCR Text ──────────────────────────────────")
    print(result["raw_text"])

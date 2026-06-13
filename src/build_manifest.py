"""
Regenerate ``tesla_financial_reports/manifest.csv`` from SEC EDGAR.

The original course pipeline (notebook ``1.WebScraping_financial_report.ipynb``)
wrote this manifest as the *source of truth for filing dates*, but the file is
not present in this project folder. The corrected ablation
(``src/corrected_ablation.py``) needs true ``filing_date`` values to fix the
look-ahead leak (Bug 2) and the weekend-drop bug (Bug 3), so we re-harvest the
filing index directly from EDGAR.

This is a faithful port of the EDGAR-listing logic from notebook 1: same CIK,
same ``submissions`` JSON endpoint (recent + historical pages), same
``WANTED_FORMS`` filter, same dedup. It does NOT download any PDFs -- it only
needs the metadata columns. No paid API is involved; EDGAR is free/public.

Run standalone:
    python src/build_manifest.py
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import requests

# --- Config (ported verbatim in spirit from notebook 1) ---------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "tesla_financial_reports"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

CIK = "0001318605"  # Tesla, Inc.
SEC_BASE = "https://data.sec.gov"
SEC_SUBMISSIONS = f"{SEC_BASE}/submissions/CIK{CIK}.json"

START_DATE = date(2010, 6, 1)
END_DATE_REQUESTED = date(2026, 12, 31)
END_DATE = min(END_DATE_REQUESTED, date.today())

SEC_PAUSE = 0.25
TIMEOUT = 25

# SEC requires a descriptive User-Agent with contact info. Override via env.
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "TextMiningProject-CorrectedAblation/1.0 (jasminejiang57@gmail.com)",
)

WANTED_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

MANIFEST_FIELDS = [
    "effective_date",
    "filing_date",
    "report_date",
    "form",
    "amended",
    "category",
    "accession",
    "primary_document",
]


@dataclass
class Filing:
    filing_date: date
    report_date: Optional[date]
    form: str
    accession: str
    primary_doc: str
    category: str
    amended: bool

    @property
    def effective_date(self) -> date:
        return self.report_date or self.filing_date


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": SEC_USER_AGENT, "Accept": "application/json"})
    return s


def _get_json(s: requests.Session, url: str) -> dict:
    last = None
    for i in range(4):
        try:
            r = s.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            time.sleep(SEC_PAUSE)
            return r.json()
        except Exception as e:  # noqa: BLE001 - mirror notebook behaviour
            last = e
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"SEC JSON fetch failed: {url} :: {last}")


def _parse_date_safe(s: Optional[str]) -> Optional[date]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date() if s else None
    except Exception:  # noqa: BLE001
        return None


def _harvest(payload: dict) -> List[Filing]:
    out: List[Filing] = []
    recent = payload.get("filings", {}).get("recent", {})
    n = len(recent.get("form", []))
    for i in range(n):
        form = str(recent["form"][i]).strip()
        if form not in WANTED_FORMS:
            continue
        filing_date = _parse_date_safe(recent["filingDate"][i])
        report_date = _parse_date_safe(recent.get("reportDate", [None] * n)[i])
        acc = str(recent["accessionNumber"][i]).strip()
        prim = str(recent.get("primaryDocument", [""] * n)[i]).strip()
        if not filing_date or not acc:
            continue
        eff = report_date or filing_date
        if not (START_DATE <= eff <= END_DATE):
            continue
        amended = form.endswith("/A")
        category = "annual" if form.startswith("10-K") else "quarterly"
        out.append(
            Filing(
                filing_date=filing_date,
                report_date=report_date,
                form=form,
                accession=acc,
                primary_doc=prim,
                category=category,
                amended=amended,
            )
        )
    return out


def load_all_filings() -> List[Filing]:
    sec = _build_session()
    root = _get_json(sec, SEC_SUBMISSIONS)
    filings = _harvest(root)

    for f in root.get("filings", {}).get("files", []) or []:
        name = f.get("name")
        if not name:
            continue
        filings.extend(_harvest(_get_json(sec, f"{SEC_BASE}/submissions/{name}")))

    seen = set()
    unique: List[Filing] = []
    for x in filings:
        key = (x.accession, x.form)
        if key in seen:
            continue
        seen.add(key)
        unique.append(x)

    unique.sort(key=lambda f: (f.effective_date, f.form), reverse=True)
    return unique


def build_manifest(out_path: Path = MANIFEST_PATH) -> Path:
    filings = load_all_filings()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for f in filings:
            w.writerow(
                {
                    "effective_date": f.effective_date.isoformat(),
                    "filing_date": f.filing_date.isoformat(),
                    "report_date": f.report_date.isoformat() if f.report_date else "",
                    "form": f.form,
                    "amended": "yes" if f.amended else "no",
                    "category": f.category,
                    "accession": f.accession,
                    "primary_document": f.primary_doc,
                }
            )
    annual = sum(f.category == "annual" for f in filings)
    quarterly = sum(f.category == "quarterly" for f in filings)
    print(
        f"[manifest] wrote {len(filings)} filings "
        f"(10-K group: {annual}, 10-Q group: {quarterly}) -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    build_manifest()

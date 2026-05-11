"""Adapter — surfaces prompts that look non-English.

Real language detection (e.g. ``langdetect``) is overkill for a heuristic
mining adapter; we use two cheap signals:

  1. Trace metadata: ``row.get("metadata", {}).get("language")`` — if the
     runtime already tagged the trace with a language code (anything but
     ``"en"`` triggers).
  2. Heuristic fallback: ratio of non-ASCII letters in the prompt is above
     a small threshold. Catches Romance / CJK / Cyrillic prompts cleanly.

Adapter yields one ``DatasetSample`` per unique non-English prompt. The
``tag`` carries the detected language code when known, or ``"non_en"``
when only the heuristic triggered.

Source label: ``"language router"`` — matches the UI's source chip.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

from router.data.dataset import DatasetSample

from .base import build_sample, embed_list, prompt_hash


SOURCE_LABEL = "language router"
_DEFAULT_TRACES_RAW = Path("traces") / "raw"

# Threshold tuned by inspection: PT/ES sentences typically carry one
# accent per 20+ letters (~4–5%). 3% catches them while still rejecting
# English with a single loanword like "café" in a long phrase.
_NON_ASCII_THRESHOLD = 0.03

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def iter_candidates(
    *,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
    embedder,
    existing: Optional[set[str]] = None,
    traces_root: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Iterator[DatasetSample]:
    seen = set(existing or ())
    root = traces_root or _DEFAULT_TRACES_RAW
    yielded = 0

    for row in _iter_raw_rows(root, since_iso, until_iso):
        prompt = (row.get("request") or "").strip()
        if not prompt:
            continue

        lang_tag = _classify(prompt, row)
        if lang_tag is None:
            continue

        sid = prompt_hash(prompt, lang_tag)
        if sid in seen:
            continue
        seen.add(sid)

        yield build_sample(
            prompt=prompt,
            tag=lang_tag,
            trace_id=row.get("trace_id"),
            source=SOURCE_LABEL,
            embedding=embed_list(embedder, prompt),
        )
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _classify(prompt: str, row: dict) -> Optional[str]:
    """Return language tag for non-EN prompts, or None when prompt is EN.

    Order of evidence:
      1. row['metadata']['language'] when present and != 'en'
      2. non-ASCII ratio heuristic → 'non_en'
    """
    meta = row.get("metadata") or {}
    lang = (meta.get("language") or "").lower()
    if lang and lang != "en":
        return lang

    letters = _LETTER_RE.findall(prompt)
    if not letters:
        return None
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    if non_ascii / max(len(letters), 1) >= _NON_ASCII_THRESHOLD:
        return "non_en"
    return None


def _iter_raw_rows(
    traces_root: Path,
    since_iso: Optional[str],
    until_iso: Optional[str],
) -> Iterator[dict]:
    if not traces_root.exists():
        return
    since_date = (since_iso or "")[:10] if since_iso else ""
    until_date = (until_iso or "")[:10] if until_iso else ""
    for path in sorted(traces_root.glob("*.jsonl")):
        date = path.stem
        if since_date and date < since_date:
            continue
        if until_date and date >= until_date:
            continue
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

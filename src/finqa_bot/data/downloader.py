"""Download FinQA splits from the official repository.

We prefer the canonical JSON files at ``github.com/czyssrs/FinQA/dataset/`` over
the HuggingFace dataset card because (a) the HF version occasionally lags
behind the GitHub copy and (b) we want the exact format expected by the
official ``evaluate.py`` script, which we replicate bit-for-bit in
:mod:`finqa_bot.eval.finqa_metric`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from finqa_bot.logging import get_logger

log = get_logger(__name__)

SPLITS: dict[str, str] = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}


@retry(
    retry=retry_if_exception_type((httpx.HTTPError,)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
    reraise=True,
)
def _fetch(url: str) -> bytes:
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def download_finqa(
    data_dir: Path | str,
    split: str = "all",
    force: bool = False,
) -> dict[str, Path]:
    """Download one or more FinQA splits into ``data_dir/raw/finqa``.

    Returns a mapping from split name to the local JSON path. Idempotent: if a
    file is present and parses as JSON, it is skipped unless ``force`` is set.
    """
    root = Path(data_dir) / "raw" / "finqa"
    root.mkdir(parents=True, exist_ok=True)
    targets: Iterable[str]
    if split == "all":
        targets = SPLITS.keys()
    elif split in SPLITS:
        targets = [split]
    else:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(SPLITS)} or 'all'.")

    paths: dict[str, Path] = {}
    for s in targets:
        dst = root / f"{s}.json"
        paths[s] = dst
        if dst.exists() and not force:
            try:
                with dst.open("r", encoding="utf-8") as f:
                    json.load(f)
                log.info("FinQA %s already present at %s (skipping).", s, dst)
                continue
            except json.JSONDecodeError:
                log.warning("Existing %s is corrupt; re-downloading.", dst)
        log.info("Downloading FinQA %s split from %s ...", s, SPLITS[s])
        blob = _fetch(SPLITS[s])
        dst.write_bytes(blob)
        log.info("Wrote %d bytes to %s.", len(blob), dst)
    return paths

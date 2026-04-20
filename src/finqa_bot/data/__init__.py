"""Data ingestion subpackage.

Public entry points:
- :func:`finqa_bot.data.downloader.download_finqa`
- :func:`finqa_bot.data.sample.load_samples`
- :func:`finqa_bot.data.chunking.chunk_record`
"""

from __future__ import annotations

from finqa_bot.data.chunking import chunk_record
from finqa_bot.data.downloader import download_finqa
from finqa_bot.data.sample import load_samples

__all__ = ["chunk_record", "download_finqa", "load_samples"]

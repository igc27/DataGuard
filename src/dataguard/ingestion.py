"""Strict, bounded CSV ingestion; never persists uploaded data."""

import csv
import io
from collections import Counter

import pandas as pd

from dataguard.config import MAX_COLUMNS, MAX_ROWS, MAX_UPLOAD_BYTES


class DatasetError(ValueError):
    """A user-facing, actionable dataset validation failure."""


def read_csv(data: bytes, *, encoding: str = "auto", delimiter: str = "auto") -> pd.DataFrame:
    """Read UTF-8/BOM or Latin-1 CSV, with explicit overrides and bounded dimensions."""
    if not data or not data.strip():
        raise DatasetError("The CSV is empty. Upload a header and at least one data row.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DatasetError("CSV exceeds the 25 MiB upload limit.")
    if b"\x00" in data:
        raise DatasetError("Binary or UTF-16 content detected. Export the CSV as UTF-8.")
    try:
        if encoding == "auto":
            try:
                decoded = data.decode("utf-8-sig")
                used_encoding = "UTF-8"
            except UnicodeDecodeError:
                decoded = data.decode("latin-1")
                used_encoding = "Latin-1 (fallback; check accented characters)"
        else:
            decoded = data.decode(encoding)
            used_encoding = encoding
        sep = delimiter
        if sep == "auto":
            try:
                sep = csv.Sniffer().sniff(decoded[:8192], delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","
        header = next(csv.reader(io.StringIO(decoded), delimiter=sep))
        if any(not name.strip() for name in header):
            raise DatasetError("Every column must have a non-empty header.")
        duplicates = [name for name, count in Counter(header).items() if count > 1]
        if duplicates:
            raise DatasetError(f"Duplicate column names: {', '.join(duplicates[:5])}. Rename them.")
        if len(header) > MAX_COLUMNS:
            raise DatasetError(f"CSV exceeds the {MAX_COLUMNS}-column limit.")
        frame = pd.read_csv(io.StringIO(decoded), sep=sep, nrows=MAX_ROWS + 1)
    except DatasetError:
        raise
    except (ValueError, UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        raise DatasetError(f"Could not parse this CSV. Check delimiter and quoting: {exc}") from exc
    if frame.empty:
        raise DatasetError("The CSV has no data rows.")
    if len(frame) > MAX_ROWS:
        raise DatasetError(f"CSV exceeds {MAX_ROWS:,} rows. Upload a representative subset.")
    frame.attrs["ingestion"] = {"encoding": used_encoding, "delimiter": repr(sep)}
    return frame

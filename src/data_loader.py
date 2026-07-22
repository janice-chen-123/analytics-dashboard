"""
data_loader.py — CSV file reading and validation.

Responsibilities:
- Validate file type and size
- Read CSV into a DataFrame
- Detect and handle encoding issues
- Detect empty files
- Return structured results so the caller can display clear error messages
"""

import io
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB


@dataclass
class LoadResult:
    """
    Result returned by load_csv().

    success: True if the file was loaded without errors.
    df:      The loaded DataFrame (empty DataFrame if success is False).
    error:   Human-readable error message shown to the user.
    warnings: Non-fatal issues (e.g. some rows had parse problems).
    """

    success: bool
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str = ""
    warnings: list[str] = field(default_factory=list)


def load_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> LoadResult:
    """
    Read an uploaded Streamlit file object into a pandas DataFrame.

    Validates file extension and size before reading.
    Tries UTF-8 first, then falls back to latin-1 for files with special characters.

    Args:
        uploaded_file: The object returned by st.file_uploader().

    Returns:
        LoadResult with success=True and a populated df, or success=False with
        an error message.
    """
    # ── Extension check ───────────────────────────────────────────────────────
    file_name: str = uploaded_file.name
    suffix = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if suffix not in ALLOWED_EXTENSIONS:
        return LoadResult(
            success=False,
            error=f"Unsupported file type '{suffix}'. Please upload a CSV file (.csv).",
        )

    # ── Size check ────────────────────────────────────────────────────────────
    raw_bytes: bytes = uploaded_file.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return LoadResult(
            success=False,
            error=(
                f"File is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit. "
                "Please upload a smaller file."
            ),
        )

    # ── Empty file check ──────────────────────────────────────────────────────
    if len(raw_bytes) == 0:
        return LoadResult(success=False, error="The uploaded file is empty.")

    # ── Read CSV ──────────────────────────────────────────────────────────────
    warnings: list[str] = []
    df: pd.DataFrame | None = None

    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
            if encoding != "utf-8":
                warnings.append(
                    f"File was read using '{encoding}' encoding instead of UTF-8. "
                    "Special characters may appear differently."
                )
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return LoadResult(
                success=False,
                error="The CSV file has no data rows. Please check the file contents.",
            )
        except pd.errors.ParserError as exc:
            return LoadResult(
                success=False,
                error=f"Could not parse the CSV file: {exc}. Please verify the file format.",
            )
        except Exception as exc:
            return LoadResult(
                success=False,
                error=f"Unexpected error while reading the file: {exc}",
            )

    if df is None:
        return LoadResult(
            success=False,
            error=(
                "Could not read the file with UTF-8 or latin-1 encoding. "
                "Try saving the file as UTF-8 and uploading again."
            ),
        )

    # ── Post-read validation ──────────────────────────────────────────────────
    if df.empty:
        return LoadResult(
            success=False,
            error="The CSV file contains column headers but no data rows.",
        )

    if len(df.columns) == 1:
        warnings.append(
            "Only one column was detected. If your file uses a semicolon (;) or tab "
            "as a separator, it may not have been parsed correctly."
        )

    return LoadResult(success=True, df=df, warnings=warnings)


@st.cache_data(show_spinner=False)
def get_file_summary(df: pd.DataFrame) -> dict:
    """
    Compute basic file statistics shown in the Data Preview section.

    Cached so repeated renders don't recompute these on every interaction.

    Args:
        df: The raw (uncleaned) DataFrame.

    Returns:
        Dict with row_count, col_count, column_names, dtypes_df.
    """
    dtypes_df = pd.DataFrame(
        {
            "Column": df.columns.tolist(),
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Sample Value": [
                str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else "N/A"
                for col in df.columns
            ],
        }
    )
    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes_df": dtypes_df,
    }

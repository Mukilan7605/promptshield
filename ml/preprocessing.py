"""
PromptShield — Dataset Preprocessing Pipeline
==============================================

Responsibilities
----------------
1. Load raw Parquet splits from datasets/raw/ (READ-ONLY — never modifies originals).
2. Validate schema, dtypes, and required columns.
3. Clean text: whitespace normalisation, control-char removal, length filtering.
4. Handle missing/invalid records with per-rule audit counters.
5. Remove exact duplicate texts within each split (configurable strategy).
6. Enrich with derived columns: label_name, category_group, severity_norm,
   text_length, word_count, text_original (preserved raw copy).
7. Save processed splits to datasets/processed/ as Parquet + CSV.
8. Emit a machine-readable JSON report covering every quality dimension.

Usage
-----
    # From project root:
    python -m ml.preprocessing                    # uses default config
    python -m ml.preprocessing --config path.yaml # custom config

The pipeline is fully deterministic — given the same raw files and config,
it always produces identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("promptshield.preprocessing")

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "preprocessing.yaml"
# Public alias — importable by tests and notebooks
DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Dataclasses for clean config access
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    train_file: str
    validation_file: str
    test_file: str
    output_train: str
    output_validation: str
    output_test: str
    output_report: str


@dataclass
class CleaningConfig:
    drop_empty_text: bool = True
    min_text_length: int = 5
    max_text_length: int = 10_000
    intra_split_dedup: str = "keep_first"


@dataclass
class PreprocessingConfig:
    paths: PathsConfig
    cleaning: CleaningConfig
    required_columns: list[str]
    output_columns: list[str]
    label_mapping: dict[int, str]
    severity_norm_map: dict[str, str]
    category_groups: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Audit counters — track every row decision
# ---------------------------------------------------------------------------

@dataclass
class SplitAudit:
    split_name: str
    rows_raw: int = 0
    rows_missing_schema: int = 0
    rows_null_text: int = 0
    rows_too_short: int = 0
    rows_too_long: int = 0
    rows_dedup_removed: int = 0
    rows_processed: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    category_distribution: dict[str, int] = field(default_factory=dict)
    category_group_distribution: dict[str, int] = field(default_factory=dict)
    severity_distribution: dict[str, int] = field(default_factory=dict)
    severity_norm_distribution: dict[str, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    text_length_stats: dict[str, float] = field(default_factory=dict)
    word_count_stats: dict[str, float] = field(default_factory=dict)
    injection_rate: float = 0.0


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> PreprocessingConfig:
    """Parse the YAML config file into a typed config object."""
    log.info(f"Loading config from: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths_raw = raw["paths"]
    paths = PathsConfig(
        raw_dir=Path(paths_raw["raw_dir"]),
        processed_dir=Path(paths_raw["processed_dir"]),
        train_file=paths_raw["train_file"],
        validation_file=paths_raw["validation_file"],
        test_file=paths_raw["test_file"],
        output_train=paths_raw["output_train"],
        output_validation=paths_raw["output_validation"],
        output_test=paths_raw["output_test"],
        output_report=paths_raw["output_report"],
    )

    cleaning_raw = raw.get("cleaning", {})
    cleaning = CleaningConfig(
        drop_empty_text=cleaning_raw.get("drop_empty_text", True),
        min_text_length=cleaning_raw.get("min_text_length", 5),
        max_text_length=cleaning_raw.get("max_text_length", 10_000),
        intra_split_dedup=cleaning_raw.get("intra_split_dedup", "keep_first"),
    )

    label_mapping = {
        int(k): v for k, v in raw["labels"]["mapping"].items()
    }

    return PreprocessingConfig(
        paths=paths,
        cleaning=cleaning,
        required_columns=raw["schema"]["required_columns"],
        output_columns=raw["schema"]["output_columns"],
        label_mapping=label_mapping,
        severity_norm_map=raw["severity"]["norm_map"],
        category_groups=raw["category_groups"],
    )


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

# Matches runs of more than one blank line
_MULTI_BLANK = re.compile(r"\n{3,}")
# Matches non-printable control chars except \t \n \r
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(raw_text: str) -> str:
    """
    Clean a single prompt text string.

    Steps (order matters):
    1. Strip leading/trailing whitespace.
    2. Remove non-printable control characters (keep \\t, \\n, \\r).
    3. Normalise Unicode to NFC (composed form).
    4. Collapse 3+ consecutive newlines to 2.
    5. Collapse runs of spaces/tabs within a line.

    The *semantics* of the text are never altered — no lowercasing,
    no punctuation removal, no stemming.  Attack strings must remain
    intact for the classifier to learn from them.
    """
    if not isinstance(raw_text, str):
        return ""

    text = raw_text.strip()
    text = _CTRL_CHARS.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _MULTI_BLANK.sub("\n\n", text)

    # Collapse intra-line whitespace runs (spaces/tabs only, not newlines)
    lines = []
    for line in text.split("\n"):
        lines.append(re.sub(r"[ \t]{2,}", " ", line).rstrip())
    text = "\n".join(lines)

    return text.strip()


def compute_sha256(text: str) -> str:
    """Return hex SHA-256 of the text string (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enrichment helpers
# ---------------------------------------------------------------------------

def resolve_category_group(category: str, groups: dict[str, list[str]]) -> str:
    """Map a fine-grained category to its high-level group name."""
    for group_name, members in groups.items():
        if category in members:
            return group_name
    return "other"


def normalise_severity(raw_severity: Any, norm_map: dict[str, str]) -> str:
    """Normalise raw severity value (may be empty string or NaN)."""
    if pd.isna(raw_severity) or raw_severity is None:
        return norm_map.get("", "none")
    return norm_map.get(str(raw_severity).strip().lower(), str(raw_severity).strip().lower())


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_split(
    df: pd.DataFrame,
    split_name: str,
    cfg: PreprocessingConfig,
) -> tuple[pd.DataFrame, SplitAudit]:
    """
    Apply the full preprocessing pipeline to a single split.

    Parameters
    ----------
    df          : Raw DataFrame loaded from Parquet.
    split_name  : One of 'train', 'validation', 'test'.
    cfg         : Parsed configuration object.

    Returns
    -------
    (processed_df, audit) where audit contains per-rule row counts.
    """
    audit = SplitAudit(split_name=split_name, rows_raw=len(df))
    log.info(f"[{split_name}] Starting — {len(df)} rows")

    # ── 1. Schema validation ───────────────────────────────────────────────
    missing_cols = [c for c in cfg.required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[{split_name}] Missing required columns: {missing_cols}"
        )

    # ── 2. Preserve original text before any cleaning ─────────────────────
    df = df.copy()
    df["text_original"] = df["text"]

    # ── 3. Clean text ──────────────────────────────────────────────────────
    df["text"] = df["text"].apply(clean_text)

    # ── 4. Drop null/empty text ───────────────────────────────────────────
    if cfg.cleaning.drop_empty_text:
        mask_empty = df["text"].str.len() == 0
        n_empty = mask_empty.sum()
        if n_empty:
            log.warning(f"[{split_name}] Dropping {n_empty} rows with empty text after cleaning")
        audit.rows_null_text = int(n_empty)
        df = df[~mask_empty].copy()

    # ── 5. Length filtering ────────────────────────────────────────────────
    min_len = cfg.cleaning.min_text_length
    max_len = cfg.cleaning.max_text_length

    too_short = df["text"].str.len() < min_len
    n_short = too_short.sum()
    if n_short:
        log.warning(f"[{split_name}] Dropping {n_short} rows below min_text_length={min_len}")
    audit.rows_too_short = int(n_short)
    df = df[~too_short].copy()

    too_long = df["text"].str.len() > max_len
    n_long = too_long.sum()
    if n_long:
        log.warning(f"[{split_name}] Dropping {n_long} rows above max_text_length={max_len}")
    audit.rows_too_long = int(n_long)
    df = df[~too_long].copy()

    # ── 6. Intra-split deduplication ──────────────────────────────────────
    strategy = cfg.cleaning.intra_split_dedup
    before_dedup = len(df)
    if strategy == "keep_first":
        df = df.drop_duplicates(subset=["text"], keep="first").copy()
    elif strategy == "keep_last":
        df = df.drop_duplicates(subset=["text"], keep="last").copy()
    elif strategy == "drop_all":
        df = df.drop_duplicates(subset=["text"], keep=False).copy()
    else:
        raise ValueError(f"Unknown intra_split_dedup strategy: {strategy!r}")
    audit.rows_dedup_removed = before_dedup - len(df)
    if audit.rows_dedup_removed:
        log.info(f"[{split_name}] Removed {audit.rows_dedup_removed} duplicate rows (strategy={strategy})")

    # ── 7. Derived columns ────────────────────────────────────────────────

    # label_name
    df["label_name"] = df["label"].map(cfg.label_mapping).fillna("unknown")

    # category_group
    df["category_group"] = df["category"].apply(
        lambda c: resolve_category_group(c, cfg.category_groups)
    )

    # severity normalisation
    df["severity_norm"] = df["severity"].apply(
        lambda s: normalise_severity(s, cfg.severity_norm_map)
    )

    # text metrics
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    # ── 8. Reset index ─────────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    audit.rows_processed = len(df)

    # ── 9. Build audit stats ───────────────────────────────────────────────
    audit.label_distribution = df["label_name"].value_counts().to_dict()
    audit.category_distribution = df["category"].value_counts().to_dict()
    audit.category_group_distribution = df["category_group"].value_counts().to_dict()
    audit.severity_distribution = df["severity"].value_counts().to_dict()
    audit.severity_norm_distribution = df["severity_norm"].value_counts().to_dict()
    audit.source_distribution = df["source"].value_counts().to_dict()
    audit.injection_rate = float(df["label"].mean())

    tl = df["text_length"]
    audit.text_length_stats = {
        "min": float(tl.min()), "max": float(tl.max()),
        "mean": round(float(tl.mean()), 2), "median": float(tl.median()),
        "p95": float(tl.quantile(0.95)), "p99": float(tl.quantile(0.99)),
    }

    wc = df["word_count"]
    audit.word_count_stats = {
        "min": float(wc.min()), "max": float(wc.max()),
        "mean": round(float(wc.mean()), 2), "median": float(wc.median()),
        "p95": float(wc.quantile(0.95)), "p99": float(wc.quantile(0.99)),
    }

    log.info(
        f"[{split_name}] Done — {audit.rows_processed} rows retained  "
        f"(injection_rate={audit.injection_rate:.1%})"
    )

    return df, audit


# ---------------------------------------------------------------------------
# Cross-split leakage check
# ---------------------------------------------------------------------------

def check_cross_split_leakage(
    splits: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """
    Check for identical prompt texts appearing in more than one split.
    Returns a dict of overlap counts for the report.
    """
    names = list(splits.keys())
    results: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            set_a = set(splits[a]["text"])
            set_b = set(splits[b]["text"])
            overlap = set_a & set_b
            key = f"{a}+{b}"
            results[key] = len(overlap)
            if overlap:
                log.warning(
                    f"DATA LEAKAGE DETECTED — {len(overlap)} texts shared between "
                    f"'{a}' and '{b}'"
                )
            else:
                log.info(f"No cross-split text overlap: {a} vs {b}")

    # group_id leakage
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            set_a = set(splits[a]["group_id"])
            set_b = set(splits[b]["group_id"])
            overlap = set_a & set_b
            key = f"group_id:{a}+{b}"
            results[key] = len(overlap)
            if overlap:
                log.warning(
                    f"GROUP ID OVERLAP — {len(overlap)} group_ids shared between "
                    f"'{a}' and '{b}'"
                )
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(
    audits: list[SplitAudit],
    leakage: dict[str, Any],
    cfg: PreprocessingConfig,
    pipeline_version: str,
) -> dict[str, Any]:
    """Assemble the full JSON report from audits and leakage results."""
    return {
        "pipeline_version": pipeline_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_cleaning": {
            "drop_empty_text": cfg.cleaning.drop_empty_text,
            "min_text_length": cfg.cleaning.min_text_length,
            "max_text_length": cfg.cleaning.max_text_length,
            "intra_split_dedup": cfg.cleaning.intra_split_dedup,
        },
        "splits": {a.split_name: _audit_to_dict(a) for a in audits},
        "cross_split_leakage": leakage,
        "label_mapping": cfg.label_mapping,
        "severity_norm_map": cfg.severity_norm_map,
        "category_groups": cfg.category_groups,
    }


def _audit_to_dict(a: SplitAudit) -> dict[str, Any]:
    return {
        "rows_raw": a.rows_raw,
        "rows_null_text": a.rows_null_text,
        "rows_too_short": a.rows_too_short,
        "rows_too_long": a.rows_too_long,
        "rows_dedup_removed": a.rows_dedup_removed,
        "rows_processed": a.rows_processed,
        "rows_dropped_total": (
            a.rows_null_text + a.rows_too_short +
            a.rows_too_long + a.rows_dedup_removed
        ),
        "injection_rate": round(a.injection_rate, 4),
        "label_distribution": a.label_distribution,
        "category_distribution": a.category_distribution,
        "category_group_distribution": a.category_group_distribution,
        "severity_distribution": a.severity_distribution,
        "severity_norm_distribution": a.severity_norm_distribution,
        "source_distribution": a.source_distribution,
        "text_length_stats": a.text_length_stats,
        "word_count_stats": a.word_count_stats,
    }


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "1.0.0"

SPLIT_MAP = [
    ("train",      "train_file",      "output_train"),
    ("validation", "validation_file", "output_validation"),
    ("test",       "test_file",       "output_test"),
]


def run_pipeline(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    """
    Execute the full preprocessing pipeline.

    Returns the report dict (also saved to disk as JSON).
    Safe to call programmatically from tests or notebooks.
    """
    cfg = load_config(config_path)

    # Ensure output directory exists
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    raw_frames: dict[str, pd.DataFrame] = {}
    processed_frames: dict[str, pd.DataFrame] = {}
    audits: list[SplitAudit] = []

    # ── Load + process each split ─────────────────────────────────────────
    for split_name, file_attr, out_attr in SPLIT_MAP:
        raw_file = cfg.paths.raw_dir / getattr(cfg.paths, file_attr)

        if not raw_file.exists():
            raise FileNotFoundError(
                f"Raw file not found: {raw_file}\n"
                f"Expected location: {raw_file.resolve()}"
            )

        log.info(f"Loading raw {split_name} split from: {raw_file}")
        df_raw = pd.read_parquet(raw_file, engine="fastparquet")
        raw_frames[split_name] = df_raw

        df_processed, audit = process_split(df_raw, split_name, cfg)
        processed_frames[split_name] = df_processed
        audits.append(audit)

        # Save processed split
        out_path = cfg.paths.processed_dir / getattr(cfg.paths, out_attr)

        # Parquet (primary — used by training pipeline)
        df_processed.to_parquet(out_path, engine="fastparquet", index=False)

        # CSV (human-readable inspection)
        csv_path = out_path.with_suffix(".csv")
        df_processed.to_csv(csv_path, index=False, encoding="utf-8")

        log.info(f"Saved processed {split_name} → {out_path}  +  {csv_path}")

    # ── Cross-split leakage ───────────────────────────────────────────────
    leakage = check_cross_split_leakage(processed_frames)

    # ── Build and save report ─────────────────────────────────────────────
    report = build_report(audits, leakage, cfg, PIPELINE_VERSION)
    report_path = cfg.paths.processed_dir / cfg.paths.output_report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Preprocessing report saved → {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PromptShield Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to preprocessing YAML config (default: {_DEFAULT_CONFIG})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = run_pipeline(config_path=args.config)

    # Pretty-print summary to console
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    for split_name, stats in report["splits"].items():
        print(
            f"  {split_name:12s}  raw={stats['rows_raw']:5d}  "
            f"processed={stats['rows_processed']:5d}  "
            f"dropped={stats['rows_dropped_total']:3d}  "
            f"injection_rate={stats['injection_rate']:.1%}"
        )
    print()
    leakage = report["cross_split_leakage"]
    leak_found = any(v > 0 for k, v in leakage.items() if not k.startswith("group_id"))
    print(f"  Cross-split text leakage: {'DETECTED ⚠' if leak_found else 'None detected OK'}")
    print("=" * 60 + "\n")

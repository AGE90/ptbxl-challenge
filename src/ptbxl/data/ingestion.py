"""Loads PTB-XL metadata, target labels, and waveform signals into a `Dataset`."""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from ptbxl.config import DataConfig
from ptbxl.utils.paths import data_raw_dir

LEADS = ["I", "II", "III", "aVF", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
QUALITY_COLUMNS = [
    "static_noise",
    "burst_noise",
    "baseline_drift",
    "electrodes_problems",
]
DEMOGRAPHIC_COLUMNS = ["age", "sex"]
TARGET_COLUMN = "target"


@dataclass
class Dataset:
    # metadata: indexed by ecg_id, includes 'target', 'strat_fold', 'patient_id'
    metadata: pd.DataFrame
    signals: np.ndarray  # (records, samples, leads), aligned to metadata row order


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Loads `ptbxl_database.csv`, parsing `scp_codes` into dicts."""
    metadata = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    metadata["patient_id"] = metadata["patient_id"].astype(int)
    metadata["scp_codes"] = metadata["scp_codes"].apply(ast.literal_eval)
    return metadata


def load_rhythm_codes(raw_dir: Path) -> set[str]:
    """SCP statement codes that denote a rhythm abnormality (excludes PACE/SR)."""
    scp_statements = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    rhythm_statements = scp_statements[
        (scp_statements["rhythm"] == 1)
        & (scp_statements.index != "PACE")
        & (scp_statements.index != "SR")
    ]
    return set(rhythm_statements.index)


def build_target(metadata: pd.DataFrame, rhythm_codes: set[str]) -> pd.DataFrame:
    """Labels records with a binary rhythm-abnormality target and drops noisy ones.

    A record is kept if it carries a rhythm statement anywhere in `scp_codes`
    (target=1), or its single most-likely statement is one of NORM/SR/PACE
    (target=0). Records flagged for any signal-quality issue are dropped.
    """
    has_rhythm_code = metadata["scp_codes"].apply(
        lambda codes: any(code in rhythm_codes for code in codes)
    )
    max_likelihood = metadata["scp_codes"].apply(
        lambda codes: max(codes, key=codes.get)
    )

    keep = has_rhythm_code | (
        max_likelihood.isin(["NORM", "SR", "PACE"]) & ~has_rhythm_code
    )
    dataset_metadata = metadata[keep].copy()
    dataset_metadata[TARGET_COLUMN] = has_rhythm_code[keep].astype(float)

    has_quality_issue = dataset_metadata[QUALITY_COLUMNS].notna().any(axis=1)
    return dataset_metadata[~has_quality_issue]


def load_signals(
    metadata: pd.DataFrame, sampling_rate: int, records_dir: Path
) -> np.ndarray:
    """Reads waveform records via `wfdb.rdsamp`, one per row of `metadata`.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must have a 'filename_lr' column (100Hz) or 'filename_hr' column
        (500Hz), matched by `sampling_rate`.
    sampling_rate : int
        100 loads 'filename_lr'; any other value loads 'filename_hr'.
    records_dir : Path
        Directory the filename columns are relative to.

    Returns
    -------
    np.ndarray
        (records, samples, leads) array, in `metadata` row order.
    """
    filenames = (
        metadata["filename_lr"] if sampling_rate == 100 else metadata["filename_hr"]
    )

    data = []
    for f in tqdm(filenames, desc="Loading records"):
        try:
            signal, _meta = wfdb.rdsamp(records_dir / f)
        except Exception as e:
            raise RuntimeError(f"Failed to load record {f}: {e}") from e
        data.append(signal)

    return np.array(data)


def load_dataset(config: DataConfig) -> Dataset:
    """Loads metadata, builds the target, and reads the corresponding signals."""
    raw_dir = data_raw_dir()
    metadata = load_metadata(raw_dir)
    rhythm_codes = load_rhythm_codes(raw_dir)
    metadata = build_target(metadata, rhythm_codes)

    # PhysioNet layout produced by scripts/data_download.sh.
    records_dir = data_raw_dir("physionet.org/files/ptb-xl/1.0.3")
    signals = load_signals(metadata, config.sampling_rate, records_dir)

    trim = int(config.trim_seconds * config.sampling_rate)
    if trim > 0:
        signals = signals[:, trim:-trim, :]

    return Dataset(metadata=metadata, signals=signals)

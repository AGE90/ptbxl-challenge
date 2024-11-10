from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


def load_raw_data(df: pd.DataFrame, sampling_rate: float, path: Path) -> np.ndarray:
    """
    Loads raw data from signal files listed in a DataFrame, based on 
    the provided sampling rate.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing metadata, including filenames for low-resolution or 
        high-resolution signal files.
        It is expected to have a column 'filename_lr' for low-resolution data and 
        'filename_hr' for high-resolution data.
    sampling_rate : float
        The sampling rate of the signals to load. If the sampling rate is 100, 
        the low-resolution files are loaded; otherwise, the high-resolution 
        files are loaded.
    path : Path
        The directory path where the signal files are stored.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row corresponds to a loaded signal from the files 
        listed in the DataFrame.
        The signals are extracted from the records in either low- or high-resolution 
        based on the sampling rate.
    """
    
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path / f) for f in df['filename_lr']]
    else:
        data = [wfdb.rdsamp(path / f) for f in df['filename_hr']]
    data = np.array([signal for signal, meta in data])
    return data

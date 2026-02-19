import os, sys, h5py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

# import utils
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{RB_folder}/utils_functions/')
from Preprocessing import *
from load_caisr_annotation_functions import load_annotation_in_original_fs

## Main CAISR prepared data loader ##
def load_prepared_data(path: str, signals: List[str] = [], fs: int = 200) -> pd.DataFrame:
    """
    Load prepared CAISR data from an .h5 file.

    Args:
        path (str): Path to the .h5 file containing the data.
        signals (list[str], optional): List of signal names to load. If empty, all signals are loaded. Defaults to [].
        fs (int, optional): Sampling frequency. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame containing the loaded signals.

    Raises:
        AssertionError: If there is a mismatch in the length of the signals or if not all requested signals are found.
    """
    # Read in signals from the .h5 file
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        vals = None
        cols = []

        for key in keys:
            subkeys = f[key].keys()
            for subkey in subkeys:
                # Filter based on requested signals
                if signals and subkey not in signals:
                    continue

                val = np.squeeze(f[key][subkey][:])
                
                # Initialize vals array if it's the first signal
                if vals is None:
                    vals = val
                else:
                    # Handle potential minor length differences due to rounding during resampling
                    max_len = max(vals.shape[0], len(val)) if vals.ndim > 1 else max(len(vals), len(val))
                    if vals.ndim == 1:
                        vals = np.pad(vals, (0, max_len - len(vals)), 'constant')
                    else:
                        vals = np.pad(vals, ((0, 0), (0, max_len - vals.shape[1])), 'constant')
                    
                    val_padded = np.pad(val, (0, max_len - len(val)), 'constant')
                    
                    if vals.ndim == 1:
                        vals = np.vstack([vals, val_padded])
                    else:
                        vals = np.vstack([vals, val_padded])
                
                cols.append(subkey)

    # Correcting shape for single signal case
    if vals.ndim > 1 and vals.shape[0] == len(cols):
        vals = vals.T

    # Create DataFrame from loaded data
    data = pd.DataFrame(vals, columns=cols)


    # Ensure all requested signals are loaded
    if signals:
        assert all(s in data.columns for s in signals), f'Not all requested signals found for recording {path}.'

    return data

def load_CAISR_output(path: str, signals_df: pd.DataFrame, csv_folder: str, fs: int, verbose: bool = False) -> pd.DataFrame:
    """
    Load CAISR output (e.g., stage and arousal labels) and append them to the signals DataFrame.

    Args:
        path (str): Path to the original .h5 file.
        signals_df (pd.DataFrame): DataFrame containing the signals data.
        csv_folder (str): Folder containing the CAISR output .csv files.
        fs (int): Sampling frequency.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the CAISR labels (stage, arousal) added.

    Raises:
        Exception: If the corresponding CAISR output .csv file is not found.
    """
    # Run over all tasks (e.g., stage, arousal)
    for task in ['stage', 'arousal']:
        # Setup corresponding CSV path for CAISR output
        tag = path.split('/')[-1].replace('.h5', f'_{task}.csv')
        csv_path = os.path.join(csv_folder, task, tag)

        # Check if the CAISR output file exists
        if not os.path.exists(csv_path):
            raise Exception(f'No matching "caisr_{task}" output found.')

        # Load CAISR labels and add to signals DataFrame
        labels = load_annotation_in_original_fs(task, path, csv_path, fs_original=fs, verbose=verbose)
        signals_df[f'{task}_CAISR'] = labels[task]

    return signals_df


## Breathing Trace Selection based on CPAP On/Off ##
def cpapOn_select_breathing_trace(signals_df: pd.DataFrame) -> Tuple[List[str], Optional[int]]:
    """
    Select the breathing trace based on whether CPAP is on or off during the sleep study.
    This version is robust to missing 'ptaf' or 'cflow' channels.

    Args:
        signals_df (pd.DataFrame): DataFrame containing the signal data, including the 'cpap_on' column.

    Returns:
        Tuple[List[str], Optional[int]]: Selected breathing trace and the split location for CPAP on/off.
    
    Raises:
        Exception: If 'cpap_on' column is not present in the signals DataFrame.
    """
    available_channels = [ch for ch in ['ptaf', 'cflow'] if ch in signals_df.columns]
    if not available_channels:
        return [], None

    split_loc = np.where(signals_df.cpap_on == 1)[0]
    
    if len(split_loc) == 0:
        split_loc = None
        selection = ['ptaf'] if 'ptaf' in available_channels else ['cflow']
    else:
        split_loc = split_loc[0]
        first_sleep_indices = np.where(np.logical_and(signals_df.stage > 0, signals_df.stage < 5))[0]
        if len(first_sleep_indices) == 0:
            # If no sleep is found, make a reasonable guess based on CPAP start time
            # If CPAP starts early, assume titration; otherwise, split-night.
            first_sleep_start = 0.2 * len(signals_df) # Assume sleep starts within first 20%
        else:
            first_sleep_start = first_sleep_indices[0]

        if split_loc < first_sleep_start:
            split_loc = None
            selection = ['cflow'] if 'cflow' in available_channels else ['ptaf']
        else:
            selection = [ch for ch in ['ptaf', 'cflow'] if ch in available_channels]
            if len(selection) < 2:
                split_loc = None

    return selection, split_loc

## Breathing Trace Selection based on Morphology ##
def morphology_select_breathing_trace(signals: pd.DataFrame, Fs: int, verbose: int) -> Tuple[List[str], Optional[int]]:
    """
    Select the breathing trace based on signal morphology and noise analysis.
    This version is robust to missing 'ptaf' or 'cflow' channels.
    
    Args:
        signals (pd.DataFrame): DataFrame containing signal data.
        Fs (int): Sampling frequency.
        verbose (int): Verbosity level for logging.

    Returns:
        Tuple[List[str], Optional[int]]: Selected breathing trace and potential split location.
    """
    available_channels = [ch for ch in ['ptaf', 'cflow'] if ch in signals.columns]
    if not available_channels:
        return [], None
    if len(available_channels) == 1:
        return available_channels, None
        
    try:
        st = np.where(signals.stage < 5)[0][0]
        end = len(signals) - np.where(np.flip(signals.stage.values) < 5)[0][0]
    except IndexError:
        if verbose == 2:
            print('No sleep detected.')
        return ['ptaf'], None

    stds = np.array([np.std(signals['ptaf']), np.std(signals['cflow'])])
    br_traces = ['ptaf', 'cflow']
    if sum(stds == 0) == 1:
        return [br_traces[np.where(stds > 0)[0][0]]], None

    noise = [check_for_noise(signals.loc[st:end, col], Fs)[0] for col in br_traces]
    if sum(noise) == 1:
        return [br_traces[np.where(np.array(noise) == False)[0][0]]], None

    window = 30 * 60 * Fs
    ptaf_std = signals['ptaf'].rolling(int(60 * Fs), center=True).std()
    cflow_std = signals['cflow'].rolling(int(60 * Fs), center=True).std()

    with np.errstate(invalid='ignore'): # Suppress warnings from dividing by zero if std is zero
        ptaf_on = (ptaf_std / np.nanmax(ptaf_std)).rolling(window, center=True).max().values > .1
        cflow_on = (cflow_std / np.nanmax(cflow_std)).rolling(window, center=True).max().values > .1

    loc1 = st + np.where(ptaf_on[st:end] == 0)[0] - window / 2
    loc2 = end - np.where(np.flip(cflow_on[st:end]) == 0)[0] + window / 2

    if len(loc1) == 0 and len(loc2) == 0:
        return ['ptaf'], None
    elif len(loc1) > 0 and len(loc2) == 0:
        loc1 = loc2 = loc1[0]
    elif len(loc2) > 0 and len(loc1) == 0:
        loc1 = loc2 = loc2[0]
    else:
        loc1, loc2 = loc1[0], loc2[0]

    if np.abs(loc2 - loc1) < Fs * 60 * 30:
        split_loc = int(np.mean([loc1, loc2]))
    else:
        return [], None

    if split_loc < 0.1 * len(signals):
        return ['cflow'], None
    elif split_loc > 0.9 * len(signals):
        return ['ptaf'], None

    return ['ptaf', 'cflow'], split_loc

## Noise Check for Breathing Signal ##
def check_for_noise(sig: np.ndarray, Fs: int) -> Tuple[bool, float, float]:
    """
    Check whether the signal is noisy using a frequency-based method.
    """
    sig = sig - np.nanmedian(sig)
    sig = np.nan_to_num(sig)

    ps = np.abs(np.fft.fft(sig))**2
    freqs = np.fft.fftfreq(sig.size, 1 / Fs)
    idx = np.argsort(freqs)

    min_freq, max_freq = 0.01, 0.5
    locs = (freqs[idx] > min_freq) & (freqs[idx] < max_freq)
    
    if not np.any(locs): return True, 0, 0

    f_range = freqs[idx][locs]
    power = ps[idx][locs]
    
    if len(power) == 0 or np.nanmax(power) == 0: return True, 0, 0

    peak = f_range[np.argmax(power)]
    percentage = len(np.where(power > 0.75 * np.nanmax(power))[0]) / len(power) * 100
    is_noise = peak < 0.1
    return is_noise, peak, percentage

## data formating ##
def setup_header(path: str, new_Fs: int, original_Fs: int, br_trace: List[str], split_loc: Optional[int]) -> Dict[str, object]:
    """
    Set up the header metadata for a sleep study recording.
    """
    hdr = {
        'newFs': new_Fs, 'Fs': original_Fs, 'test_type': 'diagnostic', 
        'rec_type': 'CAISR', 'cpap_start': split_loc,
        'patient_tag': path.split('/')[-1].split('.')[0]
    }
    if len(br_trace) == 0 or 'abd+chest' in br_trace:
        hdr['test_type'] = 'unknown'
    elif len(br_trace) == 2:
        hdr['test_type'] = 'split-night'
    elif br_trace[0] == 'cflow':
        hdr['test_type'] = 'titration'
    return hdr

######################
## main data-loader ##
######################

def load_breathing_signals_from_prepared_data(path: str, csv_folder: str, original_Fs: int = 200, new_Fs: int = 10, 
                                                channels: Optional[List[str]] = [], add_CAISR: bool = True, verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Load and preprocess breathing signals from prepared data.
    This version is robust to missing 'ptaf' or 'cflow' channels.
    """
    signals_df = load_prepared_data(path)

    if add_CAISR:
        try:
            signals_df = load_CAISR_output(path, signals_df, csv_folder, original_Fs)
        except Exception as error:
            raise Exception(f"Error loading CAISR output: {error}")
        signals_df['stage'] = signals_df['stage_CAISR']
        signals_df['arousal'] = signals_df['arousal_CAISR']
    else:
        signals_df['stage'] = signals_df.get('stage_expert_0', 0)
        signals_df['arousal'] = signals_df.get('arousal_expert_0', 0)

    signals_df['resp'] = signals_df.get('resp-h3_expert_0', signals_df.get('resp-h3_converted_1', 0))
    
    keep_cols = ['arousal', 'stage', 'resp', 'ptaf', 'airflow', 'cflow', 'spo2', 'abd', 'chest', 'cpap_on']
    signals_df = signals_df[[col for col in keep_cols if col in signals_df.columns]]

    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)

    # --- START OF DEFINITIVE FIX ---
    # PRE-NORMALIZATION QUALITY CHECK:
    # Identify and discard flat or near-flat respiratory flow channels before any selection or normalization.
    # This prevents artificial signals (like the square wave from your PDB) from being created and used.
    bad_channels = []
    for channel in ['ptaf', 'cflow']:
        if channel in signals_df.columns:
            # A plausible signal must have some variation. A tiny std indicates a flat-line.
            if signals_df[channel].std() < 1e-4:
                if verbose:
                    print(f"INFO for {path.split('/')[-1]}: Channel '{channel}' is flat and will be ignored.")
                bad_channels.append(channel)
    
    if bad_channels:
        signals_df = signals_df.drop(columns=bad_channels)
    # --- END OF DEFINITIVE FIX ---

    br_trace, split_loc = morphology_select_breathing_trace(signals_df, new_Fs, verbose)
    tag = 'signal morphology'

    if 'cpap_on' in signals_df.columns:
        br_trace_cpap, split_loc_cpap = cpapOn_select_breathing_trace(signals_df)
        # Only override morphology selection if cpap_on provides a valid trace
        if br_trace_cpap:
            br_trace, split_loc = br_trace_cpap, split_loc_cpap
            tag = 'cpap_on'
        signals_df = signals_df.drop(columns=['cpap_on'])
    else:
        if verbose == 2: print('No "cpap_on" channel was found.')
    
    if isinstance(split_loc, (int, float)):
        split_loc = int(split_loc * new_Fs / original_Fs)

    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # Final trace construction with robust fallback
    if split_loc is None:
        if br_trace and br_trace[0] in signals_df.columns:
            signals_df['breathing_trace'] = signals_df[br_trace[0]].values
        else:
            br_trace = [] # Ensure trace is empty if channel wasn't found or was discarded

        # Fallback to abd+chest if no valid flow trace was selected or if it's still flat post-normalization
        if not br_trace or signals_df[br_trace[0]].std() < 1e-4:
            if verbose: print(f"INFO for {path.split('/')[-1]}: No valid flow trace found. Using 'abd+chest' as fallback.")
            abd = signals_df['abd'].rolling(int(0.5 * new_Fs), center=True).median().fillna(0)
            chest = signals_df['chest'].rolling(int(0.5 * new_Fs), center=True).median().fillna(0)
            signals_df['breathing_trace'] = abd + chest
            br_trace = ['abd+chest']
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    # Ensure airflow is usable, otherwise copy from the final breathing trace
    if 'airflow' not in signals_df.columns or signals_df['airflow'].std() < 1e-4:
        signals_df['airflow'] = signals_df['breathing_trace']

    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)
    if verbose == 2:
        print('Study: ' + hdr['test_type'] + f' (based on "{tag}").')

    signals_df['patient_asleep'] = np.logical_and(signals_df.stage > 0, signals_df.stage < 5)

    return signals_df, hdr

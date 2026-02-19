"""
CAISR Metrics: Sleep Index Calculator
Cleaned/Refactored for Public Repository

Description:
    Calculates clinical sleep metrics (AHI, Sleep Efficiency, ArI, PLMI) 
    based on CAISR's combined output CSVs.
    
    Metrics calculated:
    - AHI (Apnea-Hypopnea Index)
    - OSAI (Obstructive Sleep Apnea Index)
    - CAI (Central Apnea Index)
    - ArI (Arousal Index)
    - LMI (Limb Movement Index)
    - PLMI (Periodic Limb Movement Index)
    - Sleep Efficiency

Usage:
    python caisr_metrics.py --input_dir /path/to/combined_csvs --output_dir /path/to/save
"""

import os
import argparse
import logging
import multiprocessing
import pandas as pd
import numpy as np
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_continuous_events(series: pd.Series, event_value: int) -> int:
    """
    Counts the number of distinct continuous events in a time-series.
    Example: [0, 1, 1, 1, 0, 1, 0] with event_value 1 returns 2 events.
    """
    arr = series.values
    # Find transitions. Pad with 0 to detect start/end at boundaries.
    transitions = np.where(np.diff(np.hstack(([0], arr == event_value, [0]))))[0]
    # Each event has a start and end transition, so divide by 2
    return len(transitions) // 2

def process_subject_file(file_name: str, base_path: str, sampling_frequency: int, logger_instance: logging.Logger) -> dict:
    """
    Processes a single subject's CSV file to calculate sleep indices.
    """
    subject_id = file_name.split('.')[0]
    # logger_instance.info(f"Processing subject: {subject_id}")
    
    try:
        file_path = os.path.join(base_path, file_name)
        df = pd.read_csv(file_path)
        
        if df.empty:
            logger_instance.warning(f"File {file_name} is empty. Skipping.")
            return None

        # --- Definitions ---
        # Sleep Stages: N3(1), N2(2), N1(3), R(4). W(5) is awake.
        sleep_stages = [1, 2, 3, 4] 
        
        # Calculate Sleep Time (in hours)
        # Note: CAISR output rows correspond to `sampling_frequency` (usually 1Hz or 200Hz depending on combine step).
        # The combine script usually upsamples to 200Hz, or outputs 1Hz. 
        # Ensure 'sampling_frequency' arg matches the data.
        sleep_rows_count = len(df[df['stage_caisr'].isin(sleep_stages)])
        sleep_hours = sleep_rows_count / sampling_frequency / 3600
        
        # Sleep Efficiency
        sleep_efficiency = (sleep_rows_count / len(df)) * 100 if len(df) > 0 else 0
        
        # --- Event Counting ---
        # Respiratory Events (1: OA, 2: CA, 3: MA, 4: HY)
        osa_count = count_continuous_events(df['resp_caisr'], 1) 
        ca_count = count_continuous_events(df['resp_caisr'], 2)
        ma_count = count_continuous_events(df['resp_caisr'], 3)
        hy_count = count_continuous_events(df['resp_caisr'], 4)
        
        # Arousal Events (1: Arousal)
        arousal_count = count_continuous_events(df['arousal_caisr'], 1)
        
        # Limb Events (1: Limb, 1: PLM)
        limb_count = count_continuous_events(df['limb_caisr'], 1)
        plm_count = count_continuous_events(df['plm_caisr'], 1)

        # --- Index Calculation ---
        # Denominator is sleep_hours (metrics are per hour of sleep)
        if sleep_hours > 0:
            AHI = (osa_count + ca_count + ma_count + hy_count) / sleep_hours
            OSAI = osa_count / sleep_hours
            CAI = ca_count / sleep_hours
            ArI = arousal_count / sleep_hours
            LMI = limb_count / sleep_hours
            PLMI = plm_count / sleep_hours
        else:
            AHI = OSAI = CAI = ArI = LMI = PLMI = 0

        return {
            'Subject': subject_id,
            'AHI': AHI,
            'ArI': ArI,
            'LMI': LMI,
            'PLMI': PLMI,
            'OSAI': OSAI,
            'CAI': CAI,
            'SleepEfficiency': sleep_efficiency,
            'TotalSleepTime_min': sleep_hours * 60
        }

    except Exception as e:
        logger_instance.error(f"Error processing {file_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate sleep metrics from CAISR output.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing combined CAISR CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results CSV.")
    parser.add_argument("--cohort_name", type=str, default="UnknownCohort", help="Name of the cohort (for filename).")
    parser.add_argument("--fs", type=int, default=200, help="Sampling frequency of the CSV rows (default: 200).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes.")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Get valid CSV files (excluding system files starting with _)
    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv') and not f.startswith('_')]
    logger.info(f"Found {len(csv_files)} files to process in {args.input_dir}")

    # Prepare partial function for mapping
    process_func = partial(process_subject_file, base_path=args.input_dir, sampling_frequency=args.fs, logger_instance=logger)
    
    results = []
    
    # Run parallel processing
    if args.num_workers > 1:
        logger.info(f"Starting parallel processing with {args.num_workers} workers...")
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            results = pool.map(process_func, csv_files)
    else:
        logger.info("Running sequentially...")
        results = [process_func(f) for f in csv_files]
    
    # Filter failures
    successful_results = [res for res in results if res is not None]
    
    if successful_results:
        columns = ['Subject', 'AHI', 'ArI', 'LMI', 'PLMI', 'OSAI', 'CAI', 'SleepEfficiency', 'TotalSleepTime_min']
        results_df = pd.DataFrame(successful_results, columns=columns)

        output_filepath = os.path.join(args.output_dir, f'CAISR_Indices_{args.cohort_name}.csv')
        results_df.to_csv(output_filepath, index=False)
        logger.info(f"Successfully processed {len(successful_results)}/{len(csv_files)} files.")
        logger.info(f"Results saved to: {output_filepath}")
    else:
        logger.warning("No results were generated.")

if __name__ == '__main__':
    # Fix for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
    
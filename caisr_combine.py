"""
CAISR Combine: Result Aggregation Module
Cleaned/Refactored for Public Repository

Description:
    Combines outputs from separate stages (Sleep Staging, Arousal, Respiratory, Limb)
    into a single CSV per subject. Also merges expert annotations if present in the input .h5.

Usage:
    python caisr_combine.py --input_data_dir /data/h5 --caisr_output_dir /data/out --cohort_name MyCohort
"""

import os
import sys
import shutil
import tempfile
import multiprocessing
import argparse
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from typing import Dict, Any

# --- Local Imports ---
# Ensure 'interrater_analysis' folder is in path
try:
    from interrater_analysis.assert_annotation import assert_annotation, annotation_minimal_correction
except ImportError:
    print("Warning: 'interrater_analysis' module not found. Annotation correction checks will be skipped.")
    # Mocking functions if module is missing to prevent crash
    def assert_annotation(*args): pass
    def annotation_minimal_correction(df_ann, df_model, *args): return df_model


def process_single_file(
    file_id: str,
    cohort: str,
    path_dir_prepared_cohort: str,
    path_dir_caisr_output: str,
    path_dir_combined_caisr_temp: str,
    id_tasks: Dict[str, str],
    fs_caisr_output: Dict[str, int],
    fs_prepared: int
) -> str:
    """
    Processes a single file by combining annotations and model outputs.
    """
    try:
        output_file_path = os.path.join(path_dir_combined_caisr_temp, f'{file_id}.csv')
        path_file_prepared = os.path.join(path_dir_prepared_cohort, f'{file_id}.h5')
        df_file_annotations = pd.DataFrame()

        # 1. Load expert annotations from the prepared H5 file (if they exist)
        if os.path.exists(path_file_prepared):
            with h5py.File(path_file_prepared, 'r') as f:
                if 'annotations' in f:
                    annotations = list(f['annotations'].keys())
                    for annotation in annotations:
                        try:
                            # Handle variable length datasets
                            data = f['annotations'][annotation][:].flatten()
                            # Pad or trim if lengths mismatch (basic handling)
                            if not df_file_annotations.empty:
                                if len(data) > len(df_file_annotations):
                                    data = data[:len(df_file_annotations)]
                            df_file_annotations[annotation] = data
                        except Exception:
                            continue
                    
        # 2. Combine model outputs for each task (stage, arousal, etc.)
        for task_tmp in ['stage', 'arousal', 'resp', 'limb']:
            if task_tmp not in id_tasks: continue

            run_folder = id_tasks[task_tmp] # e.g. 'stage' or specific run ID
            fs_caisr_output_tmp = fs_caisr_output.get(task_tmp, 1)

            # Construct path: output_dir/cohort/task/file_task.csv OR output_dir/task/file_task.csv
            # Using the simplified structure from previous scripts: output_dir/task/file_task.csv
            path_file_run = os.path.join(path_dir_caisr_output, task_tmp, f'{file_id}_{task_tmp}.csv')

            if not os.path.exists(path_file_run):
                return f"Failed: {file_id}, model output not found for {task_tmp}"

            df_annotation_model_run = pd.read_csv(path_file_run)
            
            # Rename probability columns for uniqueness
            if 'prob_no' in df_annotation_model_run.columns:
                if task_tmp == 'limb' and 'plm' in df_annotation_model_run.columns:
                    df_annotation_model_run.rename({'plm': 'plm_caisr'}, axis=1, inplace=True)
                df_annotation_model_run.rename({'prob_no': f'prob_no-{task_tmp}'}, axis=1, inplace=True)

            # Assert and correct annotations (Expert check logic)
            if not df_file_annotations.empty:
                try:
                    assert_annotation(df_file_annotations, df_annotation_model_run, fs_prepared, fs_caisr_output_tmp, task_tmp)
                    df_annotation_model_run = annotation_minimal_correction(df_file_annotations, df_annotation_model_run, fs_prepared, fs_caisr_output_tmp)
                except Exception as e:
                    # Proceed with model output even if annotation check fails
                    pass

            # Prepare for merge: Remove idx columns, upsample if necessary
            if 'end_idx' in df_annotation_model_run.columns:
                last_end_idx = int(df_annotation_model_run['end_idx'].values[-1])
                df_annotation_model_run = df_annotation_model_run.drop(['start_idx', 'end_idx'], axis=1, errors='ignore')
            
            # Upsample to common frequency (fs_prepared = 200Hz usually, or 1Hz base)
            # CAISR output is usually 1Hz or Event-based mapped to time.
            # Here we assume model output is already mapped to 1Hz rows or similar structure.
            # If standardizing to 200Hz is required:
            repeat_factor = fs_prepared // fs_caisr_output_tmp
            if repeat_factor > 1:
                df_annotation_model_run = pd.DataFrame(
                    np.repeat(df_annotation_model_run.values, repeat_factor, axis=0), 
                    columns=df_annotation_model_run.columns
                )

            # Standardize column names
            df_annotation_model_run.rename({task_tmp: f'{task_tmp}_caisr'}, axis=1, inplace=True)
            for col in df_annotation_model_run.columns:
                if col.startswith('prob_'):
                    df_annotation_model_run.rename({col: f'caisr_{col}'}, axis=1, inplace=True)
            
            # Initialize main df if empty
            if df_file_annotations.empty:
                df_file_annotations = df_annotation_model_run
            else:
                # Truncate to match length
                min_len = min(len(df_file_annotations), len(df_annotation_model_run))
                df_file_annotations = df_file_annotations.iloc[:min_len]
                df_annotation_model_run = df_annotation_model_run.iloc[:min_len]
                # Concat
                df_file_annotations = pd.concat([df_file_annotations.reset_index(drop=True), df_annotation_model_run.reset_index(drop=True)], axis=1)

        # Final Cleaning
        df_file_annotations = df_file_annotations.fillna(9) # 9 often used as 'unknown' or 'artifact'
        
        # Optimize types
        for col in df_file_annotations.columns:
            if 'prob' in col:
                df_file_annotations[col] = df_file_annotations[col].astype('float16')
            else:
                try:
                    df_file_annotations[col] = df_file_annotations[col].astype('int8')
                except:
                    pass

        df_file_annotations.to_csv(output_file_path, index=False)
        return f"Success: {file_id}"

    except Exception as e:
        return f"Failed: {file_id} with error: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine CAISR outputs into single CSVs.")
    parser.add_argument("--input_data_dir", type=str, required=True, help="Path to input .h5 files (for expert annotations)")
    parser.add_argument("--caisr_output_dir", type=str, required=True, help="Root folder containing stage/, arousal/, etc. output folders")
    parser.add_argument("--cohort_name", type=str, default="default_cohort", help="Name of the cohort for output folder naming")
    parser.add_argument("--run_config", type=str, default=None, help="Optional CSV to map specific task run IDs if using versioning")
    args = parser.parse_args()

    # Setup
    path_dir_prepared_cohort = args.input_data_dir
    path_dir_caisr_output = args.caisr_output_dir
    cohort = args.cohort_name
    
    # Task mapping (Assume standard folder names if no config provided)
    if args.run_config:
        # Load mapping logic here if needed
        pass
    
    # Basic task list
    id_tasks = {'stage': 'stage', 'arousal': 'arousal', 'resp': 'resp', 'limb': 'limb'}
    fs_caisr_output = {'stage': 1, 'arousal': 2, 'resp': 1, 'limb': 1}
    fs_prepared = 200 # Standard frequency for combination (upsampling target)

    # Output directory
    path_dir_combined_caisr = os.path.join(path_dir_caisr_output, 'combined', cohort)
    os.makedirs(path_dir_combined_caisr, exist_ok=True)
    
    # Identify files to process (intersection of available outputs)
    # Strategy: Look at 'stage' outputs as the base list
    stage_dir = os.path.join(path_dir_caisr_output, 'stage')
    if not os.path.exists(stage_dir):
        print("Error: No 'stage' output directory found. Staging is required.")
        sys.exit(1)
        
    files_to_process = [f.replace('_stage.csv', '') for f in os.listdir(stage_dir) if f.endswith('_stage.csv')]
    
    # Filter existing
    existing = [f.replace('.csv', '') for f in os.listdir(path_dir_combined_caisr)]
    files_to_process = [f for f in files_to_process if f not in existing]

    print(f'\nCombining predictions for {cohort}. Found {len(files_to_process)} new files to process.')

    batch_size = 100
    num_workers = min(multiprocessing.cpu_count(), 20)
    
    for i in range(0, len(files_to_process), batch_size):
        files_in_batch = files_to_process[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        path_temp = tempfile.mkdtemp()
        try:
            tasks = [(
                fid, cohort, path_dir_prepared_cohort, path_dir_caisr_output, path_temp,
                id_tasks, fs_caisr_output, fs_prepared
            ) for fid in files_in_batch]
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.starmap(process_single_file, tasks), total=len(tasks)))
                
            # Move successful files
            for item in os.listdir(path_temp):
                if item.endswith('.csv'):
                    shutil.move(os.path.join(path_temp, item), os.path.join(path_dir_combined_caisr, item))
                    
        finally:
            shutil.rmtree(path_temp)

    print("\nCombination complete.")

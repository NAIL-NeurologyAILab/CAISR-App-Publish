"""
CAISR Inter-Rater Analysis (IRA)
Cleaned/Refactored for Public Repository

Description:
    Performs inter-rater comparison between CAISR predictions and Expert annotations.
    Generates:
    1. Summary Statistics Heatmap: Displays Mean (SD) for Accuracy, Kappa, and AC1 across subjects.
    2. Normalized Confusion Matrices.
    3. ROC and Precision-Recall Curves (if probabilities available).
    4. Per-subject Kappa distribution plots.

    Uses Ray for parallel processing.

Usage:
    python caisr_interrater.py stage resp --cohort_name MyCohort \
        --caisr_dir /path/to/caisr/combined \
        --expert_dir /path/to/original/h5_files \
        --output_dir /path/to/results
"""

import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import warnings
import random
import seaborn as sns
import matplotlib.colors as mcolors
import ray

# --- Check Dependencies ---
try:
    from pycm import ConfusionMatrix
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
except ImportError:
    sys.exit("Missing dependencies. Please run: pip install pycm scikit-learn ray seaborn")

# Initialize Ray (Limit CPUs to prevent OOM)
ray.init(num_cpus=4, ignore_reinit_error=True, include_dashboard=False)
plt.rcParams['font.size'] = 9

# ----------------------------------------------------------------------------
# Section 1: Helper Functions (Visualization & Stats)
# ----------------------------------------------------------------------------

def make_colormap(seq):
    """Creates a custom linear segment colormap."""
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]; r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2]); cdict['green'].append([item, g1, g2]); cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

@ray.remote
def _calculate_cm_for_subject(subject_tuple, task_vars):
    """Ray remote worker to calculate ConfusionMatrix for a single subject."""
    study_id, df_subject = subject_tuple
    scorer_names = task_vars['scorer_names']
    class_ints = task_vars['class_ints']
    
    if len(df_subject) > 1:
        try:
            # Rename for clarity
            rename_dict = {f"{task_vars['task']}_{scorer}": scorer for scorer in scorer_names}
            df_subject = df_subject.rename(columns=rename_dict)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cm = ConfusionMatrix(
                    actual_vector=df_subject[scorer_names[1]].values.astype(float), # Expert
                    predict_vector=df_subject[scorer_names[0]].values.astype(float), # CAISR
                    classes=class_ints
                )
            return (study_id, cm)
        except Exception: 
            return (study_id, None)
    return (study_id, None)

def compute_cm_per_subject(ira_data, task_vars):
    """Distributes confusion matrix calculation across subjects using Ray."""
    print("Precomputing confusion matrices per subject (in parallel)...")
    subject_data_list = list(ira_data.groupby('study_id'))
    
    task_vars_id = ray.put(task_vars)
    futures = [_calculate_cm_for_subject.remote(s, task_vars_id) for s in subject_data_list]
    
    results = [ray.get(f) for f in tqdm(futures, desc="Per-subject CMs")]
    cms_per_subject = {study_id: cm for study_id, cm in results if cm is not None}
    
    return cms_per_subject

def get_population_stats(cms_per_subject):
    """
    Calculates Mean and SD across all subjects for key metrics.
    """
    metrics = {'Overall ACC': [], 'Gwet AC1': [], 'Kappa': []}
    
    for cm in cms_per_subject.values():
        for m in metrics.keys():
            val = cm.overall_stat.get(m, "None")
            # Filter out strings/None/NaN
            try:
                val = float(val)
                if not np.isnan(val):
                    metrics[m].append(val)
            except (ValueError, TypeError):
                continue
                
    results = {}
    for m in metrics:
        data = np.array(metrics[m])
        if len(data) > 0:
            results[m] = (np.mean(data), np.std(data))
        else:
            results[m] = (np.nan, np.nan)
            
    return results

def plot_summary_stats(ira_data, cms_per_subject, task_vars, plot_vars):
    """Calculates Mean (SD) across subjects and plots heatmaps."""
    if not cms_per_subject: return

    # 1. Calculate Population Statistics (Mean/SD)
    stats = get_population_stats(cms_per_subject)
    
    metrics_map = ['Overall ACC', 'Gwet AC1', 'Kappa']
    scorer_names = task_vars['scorer_names']
    
    # Create DataFrames for plotting
    # Structure: 2x2 matrix where [0,1] contains the metric
    dfs_point = [] # Holds mean for color mapping
    annot_dfs = [] # Holds "Mean\n(SD)" string
    
    for m in metrics_map:
        mean_val, sd_val = stats[m]
        
        # DataFrame for heatmap color (Mean)
        df_mean = pd.DataFrame([[np.nan, mean_val], [mean_val, np.nan]], 
                               index=scorer_names, columns=scorer_names)
        dfs_point.append(df_mean)
        
        # Annotation String
        val_str = f"{mean_val:.2f}\n({sd_val:.2f})"
        df_str = pd.DataFrame([[np.nan, val_str], [val_str, np.nan]], 
                              index=scorer_names, columns=scorer_names)
        
        # Clear diagonal and NaN entries
        for r in range(len(df_str)): df_str.iloc[r, r] = ''
        annot_dfs.append(df_str)

    # 2. Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), sharey=True)
    titles = ['Accuracy', 'Gwet AC1', "Cohen's Kappa"]
    cmap = make_colormap([(0,0,0), mcolors.ColorConverter().to_rgb('white')])
    
    for i, ax in enumerate(axes):
        sns.heatmap(dfs_point[i].astype(float), cmap=cmap, ax=ax, 
                    annot=annot_dfs[i], fmt='s', cbar=False, vmin=0, vmax=1)
        ax.set_title(titles[i])
    
    plt.suptitle(f"Agreement Coefficients (Mean ± SD) - {task_vars['task_full_name']}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_vars['path_savedir'], f"{task_vars['task']}_SummaryStats.pdf"))
    plt.close()

def plot_normalized_cm(ira_data, task_vars, plot_vars):
    """Plots row-normalized confusion matrix (aggregated across all data)."""
    y_true = ira_data[f"{task_vars['task']}_expert_0"]
    y_pred = ira_data[f"{task_vars['task']}_caisr"]
    
    cm = ConfusionMatrix(actual_vector=y_true.values, predict_vector=y_pred.values, classes=task_vars['class_ints'])
    cm_df = pd.DataFrame(cm.table).T.reindex(index=task_vars['class_ints'], columns=task_vars['class_ints'], fill_value=0)
    
    # Normalize
    row_counts = cm_df.sum(axis=1)
    cm_norm = cm_df.div(row_counts, axis=0).fillna(0) * 100
    cm_norm = cm_norm.round(0).astype(int)
    
    # Rename
    names = task_vars['class_names'][::-1] if task_vars['task'] == 'stage' else task_vars['class_names']
    map_dict = {v: k for k, v in task_vars['class_dict'].items()}
    cm_norm = cm_norm.rename(index=map_dict, columns=map_dict).reindex(index=names, columns=names, fill_value=0)

    # Labels
    ylabels = [f"{n} (N={int(row_counts.get(task_vars['class_dict'][n], 0))})" for n in names]

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = make_colormap([(0,0,0), mcolors.ColorConverter().to_rgb('white')])
    sns.heatmap(cm_norm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=True, vmin=0, vmax=100, xticklabels=names, yticklabels=ylabels)
    
    ax.set_title(f"Normalized Confusion Matrix\n{task_vars['task_full_name']}")
    ax.set_xlabel('CAISR Prediction'); ax.set_ylabel('Expert Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_vars['path_savedir'], f"{task_vars['task']}_CM.pdf"))
    plt.close()

def plot_roc_prc(ira_data, task_vars, plot_vars):
    """Plots ROC and PRC curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    y_true = ira_data[f"{task_vars['task']}_expert_0"]
    y_prob = ira_data[task_vars['cols_probability']]
    
    if task_vars['task'] == 'stage':
        # One-vs-Rest
        for i, cls in enumerate(task_vars['class_names_order']):
            cls_int = task_vars['class_dict'][cls]
            bin_true = (y_true == cls_int).astype(int)
            bin_prob = y_prob.iloc[:, i]
            
            fpr, tpr, _ = roc_curve(bin_true, bin_prob)
            prec, rec, _ = precision_recall_curve(bin_true, bin_prob)
            
            ax1.plot(fpr, tpr, label=f'{cls} (AUC={auc(fpr, tpr):.2f})')
            ax2.plot(rec, prec, label=f'{cls} (AUC={auc(rec, prec):.2f})')
    else:
        # Binary
        pos_int = task_vars['label_dict']['pos_labels'][0]
        bin_true = (y_true == pos_int).astype(int)
        bin_prob = y_prob.iloc[:, 1] # Prob of class 1
        
        fpr, tpr, _ = roc_curve(bin_true, bin_prob)
        prec, rec, _ = precision_recall_curve(bin_true, bin_prob)
        
        ax1.plot(fpr, tpr, label=f'AUROC = {auc(fpr, tpr):.2f}')
        ax2.plot(rec, prec, label=f'AUPRC = {auc(rec, prec):.2f}')

    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_title('ROC'); ax1.legend()
    ax2.set_title('PRC'); ax2.legend()
    plt.suptitle(f"Performance Curves: {task_vars['task_full_name']}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_vars['path_savedir'], f"{task_vars['task']}_ROC_PRC.pdf"))
    plt.close()

# ----------------------------------------------------------------------------
# Section 2: Data Loading
# ----------------------------------------------------------------------------

def process_single_session_loader(args):
    """Helper to load a single session's CSV and H5 data."""
    row, task, task_info, caisr_dir, expert_dir, downsample = args
    try:
        # Determine ID
        bids = row.get('BidsFolder') or row.get('BIDSFolder')
        study_id = f"{bids}_ses-{row['SessionID']}"
        
        # Paths
        c_path = os.path.join(caisr_dir, f"{study_id}.csv")
        e_path = os.path.join(expert_dir, f"{study_id}.h5")
        
        if not (os.path.exists(c_path) and os.path.exists(e_path)): return None
        
        # Load CAISR
        cols = [task_info['caisr_col']] + task_info.get('prob_cols', [])
        try:
            df = pd.read_csv(c_path, usecols=cols)
        except ValueError:
            # Fallback if prob cols missing
            df = pd.read_csv(c_path, usecols=[task_info['caisr_col']])

        # Load Expert
        with h5py.File(e_path, 'r') as f:
            y_exp = f['annotations'][task_info['expert_key']][:].flatten().astype(int)

        # Align length
        n = min(len(df), len(y_exp))
        df, y_exp = df.iloc[:n], y_exp[:n]

        # Expert Remapping
        if task == 'resp':
            # Remap AASM codes to simplified classes
            y_exp = pd.Series(y_exp).replace({8: 1, 4: 4, 5: 4, 6: 4, 9: 4, 7: 5, 10: 1, 11: 4}).values
        elif task == 'limb':
            y_exp[y_exp > 0] = 1 # Binary

        df[f'{task}_expert_0'] = y_exp
        
        # Downsample
        df = df.iloc[::downsample].copy()
        df['study_id'] = study_id
        return df

    except Exception:
        return None

def load_data(cohort, task, session_list, caisr_dir, expert_dir):
    """Loads and combines data for all sessions."""
    # Config
    task_map = {
        'stage':   {'caisr_col': 'stage_caisr',   'expert_key': 'stage',   'prob_cols': ['caisr_prob_w', 'caisr_prob_r', 'caisr_prob_n1', 'caisr_prob_n2', 'caisr_prob_n3']},
        'resp':    {'caisr_col': 'resp_caisr',    'expert_key': 'resp'},
        'arousal': {'caisr_col': 'arousal_caisr', 'expert_key': 'arousal', 'prob_cols': ['caisr_prob_no-arousal', 'caisr_prob_arousal']},
        'limb':    {'caisr_col': 'limb_caisr',    'expert_key': 'limb'},
    }
    
    downsample = 200 if task in ['resp', 'arousal', 'limb'] else 6000 # 30s for stage
    
    # Prepare jobs
    jobs = [(row, task, task_map[task], caisr_dir, expert_dir, downsample) for _, row in session_list.iterrows()]
    
    # Run
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as exc:
        results = list(tqdm(exc.map(process_single_session_loader, jobs), total=len(jobs), desc=f"Loading {task}"))
    
    dfs = [r for r in results if r is not None]
    if not dfs: return pd.DataFrame(), 0
    
    return pd.concat(dfs, ignore_index=True), downsample

# ----------------------------------------------------------------------------
# Section 3: Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CAISR Inter-Rater Analysis.")
    parser.add_argument("tasks", nargs='+', choices=['stage', 'resp', 'arousal', 'limb'])
    parser.add_argument("--cohort_name", required=True)
    parser.add_argument("--caisr_dir", required=True, help="Path to 'combined' CSV files")
    parser.add_argument("--expert_dir", required=True, help="Path to original .h5 files")
    parser.add_argument("--metadata_csv", help="Optional: Path to metadata CSV to filter sessions")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dev", action="store_true", help="Limit to 20 sessions for testing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Session List Generation
    if args.metadata_csv:
        df_meta = pd.read_csv(args.metadata_csv)
        if 'SessionID' in df_meta.columns and 'BidsFolder' in df_meta.columns:
            session_list = df_meta
        else:
            print("Metadata CSV missing 'SessionID' or 'BidsFolder'. Processing all files in caisr_dir.")
            session_list = pd.DataFrame([{'BidsFolder': f.split('_ses-')[0], 'SessionID': f.split('_ses-')[1].replace('.csv', '')} 
                                         for f in os.listdir(args.caisr_dir) if '_ses-' in f and f.endswith('.csv')])
    else:
        # Fallback: list files in CAISR dir
        session_list = pd.DataFrame([{'BidsFolder': f.split('_ses-')[0], 'SessionID': f.split('_ses-')[1].replace('.csv', '')} 
                                     for f in os.listdir(args.caisr_dir) if '_ses-' in f and f.endswith('.csv')])

    if args.dev: session_list = session_list.head(20)

    for task in args.tasks:
        print(f"\n--- Processing {task.upper()} ---")
        
        # Configs
        configs = {
            'stage': {'cn': ['N3', 'N2', 'N1', 'R', 'W'], 'cd': {'N3': 0, 'N2': 1, 'N1': 2, 'R': 3, 'W': 4}, 'tfn': 'Sleep Staging'},
            'resp': {'cn': ['No', 'OA', 'CA', 'MA', 'HY', 'RERA'], 'cd': {'No': 0, 'OA': 1, 'CA': 2, 'MA': 3, 'HY': 4, 'RERA': 5}, 'tfn': 'Respiratory Event Detection'},
            'arousal': {'cn': ['No-Arousal', 'Arousal'], 'cd': {'No-Arousal': 0, 'Arousal': 1}, 'tfn': 'Arousal', 'ld': {'pos_labels': [1]}},
            'limb': {'cn': ['No-Limb', 'Limb'], 'cd': {'No-Limb': 0, 'Limb': 1}, 'tfn': 'Limb Movement'}
        }
        
        cfg = configs[task]
        task_vars = {
            'task': task, 'task_full_name': cfg['tfn'], 
            'class_names': cfg['cn'], 'class_dict': cfg['cd'], 
            'class_ints': list(cfg['cd'].values()),
            'scorer_names': ['caisr', 'expert_0']
        }
        if task == 'stage': task_vars['class_names_order'] = cfg['cn'][::-1]
        if 'ld' in cfg: task_vars['label_dict'] = cfg['ld']
        if task in ['stage', 'arousal']: 
            task_vars['cols_probability'] = ['caisr_prob_w', 'caisr_prob_r', 'caisr_prob_n1', 'caisr_prob_n2', 'caisr_prob_n3'] if task=='stage' else ['caisr_prob_no-arousal', 'caisr_prob_arousal']

        # Load
        df_all, fs = load_data(args.cohort_name, task, session_list, args.caisr_dir, args.expert_dir)
        if df_all.empty:
            print(f"Skipping {task} (No data found).")
            continue

        # Filter NaNs/Artifacts
        if task != 'resp': df_all = df_all.replace(9, np.nan).dropna()
        if task == 'stage': 
             if df_all[f'{task}_expert_0'].max() > 4: 
                 df_all[f'{task}_expert_0'] -= 1
                 df_all[f'{task}_caisr'] -= 1

        plot_vars = {'path_savedir': args.output_dir}

        # Analyze
        cms = compute_cm_per_subject(df_all, task_vars)
        if not cms: continue
        
        plot_summary_stats(df_all, cms, task_vars, plot_vars)
        plot_normalized_cm(df_all, task_vars, plot_vars)
        
        if 'cols_probability' in task_vars:
             if all(c in df_all.columns for c in task_vars['cols_probability']):
                 plot_roc_prc(df_all, task_vars, plot_vars)

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
    
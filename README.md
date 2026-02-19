
# CAISR-App: Complete AI Sleep Report (Source Code)

**CAISR-App** is a comprehensive, modular pipeline for automated sleep analysis. It processes physiological signals (PSG) to detect Sleep Stages, Arousals, Respiratory Events, and Limb Movements.

> **Note:** This repository contains the **source code** used to build the CAISR Docker images. It is intended for researchers and developers who wish to customize the algorithms, retrain models, or run the pipeline natively without Docker. For the pre-built "plug-and-play" Docker solution, please refer to our main website.

## 📋 Table of Contents
- [Prerequisites](#-prerequisites)
- [Directory Structure](#-directory-structure)
- [Input Data Requirements](#-input-data-requirements)
- [Usage Pipeline](#-usage-pipeline)
- [Outputs & Metrics](#-outputs--metrics)
- [Configuration](#-configuration)
- [Authors & Contact](#-authors--contact)

---

## 🛠 Prerequisites

## 🛠 Prerequisites & Environment Setup

Because CAISR is a modular system adapted from distinct Docker containers, each module (Stage, Arousal, Resp, Limb) operates on specific dependencies. To ensure reproducibility, **you must create separate Conda environments for each task.**

**1. Install Anaconda or Miniconda**
Ensure you have [Conda](https://docs.conda.io/en/latest/) installed.

**2. Create Environments**

Run the following commands to set up the specific environments.

#### A. Sleep Staging (`caisr_stage`)
The staging module uses a specific environment file located in the `stage/` folder.
```bash
cd stage
conda env create -f caisr_stage.yml
# This creates an environment named 'caisr_stage'
cd ..
```

#### B. Arousal Detection (`caisr_arousal`)
```bash
conda create -n caisr_arousal python=3.9 -y
conda activate caisr_arousal
pip install -r arousal/arousal_requirements.txt
conda deactivate
```

#### C. Respiratory Analysis (`caisr_resp`)
```bash
conda create -n caisr_resp python=3.9 -y
conda activate caisr_resp
pip install -r resp/resp_requirements.txt
conda deactivate
```

#### D. Limb Movement (`caisr_limb`)
```bash
conda create -n caisr_limb python=3.9 -y
conda activate caisr_limb
pip install -r limb/limb_requirements.txt
conda deactivate
```

#### E. Analysis & Metrics (`caisr_analysis`)
For the `combine`, `metrics`, and `interrater` scripts.
```bash
conda create -n caisr_analysis python=3.10 -y
conda activate caisr_analysis
pip install -r requirements_analysis.txt
conda deactivate
```

---

## 📂 Directory Structure

The system requires the following structure to function correctly:

```text
CAISR-App/
├── README.md
├── requirements_analysis.txt
├── caisr_stage.py          # Sleep Staging Script
├── caisr_arousal.py        # Arousal Detection Script
├── caisr_resp.py           # Respiratory Analysis Script
├── caisr_limb.py           # Limb Movement Script
├── caisr_combine.py        # Results Aggregator
├── caisr_metrics.py        # Clinical Metrics Calculator
├── caisr_interrater.py     # Comparison Tool
├── data/                   # Run Parameters (.csv)
├── stage/                  # Staging Model Utils
├── arousal/                # Arousal Model Utils
├── resp/                   # Respiratory Rules
└── limb/                   # Limb Logic
```

---

## 💾 Input Data Requirements

This source code pipeline expects pre-processed **HDF5 (.h5)** files.

### H5 File Specification
If you are converting from EDF, your `.h5` files must match this format **exactly**.

*   **Sampling Rate**: 200 Hz (Data will be processed at this rate).
*   **Attributes**: `f.attrs['sampling_rate'] = 200`.
*   **Required Channels** (Keys in `signals/` group):
    ```text
    'f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1',
    'e1-m2', 'e2-m1', 'chin1-chin2', 'abd', 'chest', 'spo2', 'ecg', 'lat', 'rat'
    ```
*   **Optional Channels**:
    ```text
    'airflow', 'cpap_on', 'hr', 'position'
    ```

### Python Snippet for H5 Creation
Use the following function to save your dataframe/signals into the compatible format:

```python
import h5py

def save_prepared_data(path, signal_dataframe):
    """
    path: destination path (e.g., 'subject01.h5')
    signal_dataframe: pandas DataFrame where columns match the required channel names
    """
    with h5py.File(path, 'w') as f:
        f.attrs['sampling_rate'] = 200
        f.attrs['unit_voltage'] = 'V'
        group_signals = f.create_group('signals')
        for name in signal_dataframe.columns:
            # Ensure data is float32 and compressed
            group_signals.create_dataset(
                name, 
                data=signal_dataframe[name], 
                shape=(len(signal_dataframe), 1), 
                maxshape=(len(signal_dataframe), 1), 
                dtype='float32', 
                compression="gzip"
            )
```

---

## 🚀 Usage

**Important:** You must activate the corresponding Conda environment before running a specific task.

### 1. Sleep Staging
**Environment:** `caisr_stage`
```bash
conda activate caisr_stage
python caisr_stage.py \
  --input_data_dir ./my_dataset/h5 \
  --output_csv_dir ./caisr_output \
  --model_dir ./stage/models \
  --param_dir ./data
conda deactivate
```

### 2. Arousal Detection
**Environment:** `caisr_arousal`
```bash
conda activate caisr_arousal
python caisr_arousal.py \
  --input_data_dir ./my_dataset/h5 \
  --output_csv_dir ./caisr_output \
  --param_dir ./data
conda deactivate
```

### 3. Respiratory Analysis
**Environment:** `caisr_resp`
```bash
conda activate caisr_resp
python caisr_resp.py \
  --input_data_dir ./my_dataset/h5 \
  --output_csv_dir ./caisr_output \
  --param_dir ./data
conda deactivate
```

### 4. Limb Movement Detection
**Environment:** `caisr_limb`
```bash
conda activate caisr_limb
python caisr_limb.py \
  --input_data_dir ./my_dataset/h5 \
  --output_csv_dir ./caisr_output \
  --param_dir ./data
conda deactivate
```

### 5. Combining Results
**Environment:** `caisr_analysis`
Aggregates the individual outputs into one CSV per subject.
```bash
conda activate caisr_analysis
python caisr_combine.py \
  --input_data_dir ./my_dataset/h5 \
  --caisr_output_dir ./caisr_output \
  --cohort_name MyStudyCohort
```

### 6. Clinical Metrics
**Environment:** `caisr_analysis`
```bash
python caisr_metrics.py \
  --input_dir ./caisr_output/combined/MyStudyCohort \
  --output_dir ./caisr_results \
  --cohort_name MyStudyCohort
```

### 7. Inter-Rater Analysis
**Environment:** `caisr_analysis`
```bash
python caisr_interrater.py stage resp\
  --cohort_name MyStudyCohort \
  --caisr_dir /path/to/caisr/combined \
  --expert_dir /path/to/original/h5_files \
  --output_dir /path/to/results
```

---

## 📊 Outputs & Metrics

### 1. Unified CSV (`caisr_combine.py`)
Located in `./caisr_output/combined/`, this file contains time-synchronized predictions (typically at 1Hz or 200Hz) for all tasks:
*   `stage_caisr`: 0=N3, 1=N2, 2=N1, 3=R, 4=W
*   `resp_caisr`: 0=Normal, 1=Obstructive, 2=Central, 3=Mixed, 4=Hypopnea, 5=RERA
*   `arousal_caisr`: 0=No, 1=Arousal
*   `limb_caisr`: 0=No, 1=Limb Movement

### 2. Clinical Metrics (`caisr_metrics.py`)
Run the metrics script to generate a summary CSV (`caisr_sleep_metrics_all_studies.csv`) containing:

*   **Total Sleep Metrics**:
    *   `TST (h)`: Total Sleep Time
    *   `Eff (%)`: Sleep Efficiency
*   **Sleep Stage Distribution**:
    *   `N1 (%)`, `N2 (%)`, `N3 (%)`, `REM (%)`
*   **Sleep Disruption**:
    *   `WASO (min)`: Wake After Sleep Onset
    *   `SL (min)`: Sleep Latency
*   **Respiratory**:
    *   `AHI`: Apnea-Hypopnea Index
    *   `OAI`, `CAI`, `MAI`, `HYI`: Individual indices for Obstructive, Central, Mixed, and Hypopneas.
*   **Other**:
    *   `Arousal I.`: Arousal Index
    *   `LMI`: Limb Movement Index

---

## ⚙️ Configuration

The `/data` folder contains CSV configuration files (`stage.csv`, `arousal.csv`, etc.). Use these to control execution behavior:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `overwrite` | `True`/`False` | If `True`, re-processes files that already have output CSVs. |
| `multiprocess` | `True`/`False` | If `True`, enables parallel processing (CPU). |

---

## 👥 Authors & Contact

This project is free to use for non-commercial purposes. For commercial use, please contact us directly.

**CAISR Development Team:**
*   **Thijs-Enagnon Nassi, PhD** (Respiratory)
*   **Wolfgang Ganglberger, PhD**
*   **Erik-Jan Meulenbrugge** (Arousal, IRA)
*   **Samaneh Nasiri, PhD** (Staging, Limb)
*   **Haoqi Sun, PhD**
*   **Robert J Thomas, MD**
*   **M Brandon Westover, MD, PhD**
*   **Shenghan Wen**

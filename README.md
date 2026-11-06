# Resilience

Research code for exploring resilience and security in healthcare-related cyber-physical systems.

This repository currently contains experimental scripts from **two related but distinct research strands**:

1. **CPST-MalariaNet forecasting experiment** — a Burundi malaria forecasting workflow using statistical and machine-learning/deep-learning models. The main script is `srcCode/data_loader.py`.
2. **IoMT / healthcare cyber-physical system (HCPS) security analysis** — scripts for analysing network attack categories, physiological indicators, classification performance, and patient-outcome impact using the WUSTL-EHMS-2020 dataset.

> **Scope note:** These are research scripts, not yet a unified software package or production-ready clinical/security system. The two strands use different datasets and should be run independently. Review and validate each experiment before interpreting results.

## Repository contents

```text
Resilience/
├── README.md
├── requirements.txt
├── .gitignore
└── srcCode/
    ├── data_loader.py          # CPST-MalariaNet forecasting experiment
    ├── models.py               # ML model comparison for attack detection
    ├── metrics.py              # Classification metrics, ROC, confusion matrix
    ├── features.py             # Attack-category frequency analysis
    ├── plots.py                # Network/physiological feature plots
    ├── pipeline.py             # POI, PSO and ESD scenario calculations
    ├── crooslayer.txt          # Python source: cross-layer model visualisation
    ├── Layer.txt               # Python source: layer model visualisation
    ├── patientoutcome.txt      # Python source: patient-outcome scenarios
    ├── trafic Attack.txt       # Python source: traffic/attack analysis
    └── trainSPO.txt            # Python source: vital-sign analysis by temperature
```

Some files ending in `.txt` contain Python code. Rename them to `.py` before executing them as Python scripts (for example, `patientoutcome.txt` to `patient_outcome.py`). `crooslayer.txt` and `Layer.txt` appear to contain overlapping layer-visualisation code; compare them before maintaining both.

## Requirements

- Python 3.9 or later (TensorFlow compatibility may require a specific Python version; see the TensorFlow installation guidance for your platform).
- Packages listed in [`requirements.txt`](requirements.txt).

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data requirements

The datasets are **not included** in this repository. Obtain them from their legitimate source or use the files supplied under the applicable data-use terms.

| Experiment | Expected input file | Used by |
|---|---|---|
| CPST-MalariaNet | `BdiDATASETS.xlsx` | `srcCode/data_loader.py` |
| IoMT / HCPS security analysis | `wustl-ehms-2020_with_attacks_categories.csv` | `models.py`, `metrics.py`, `features.py`, `plots.py`, `pipeline.py` and related analysis scripts |

Place each input file in the working directory expected by its script, or update the corresponding file path in the script. Do not commit sensitive, restricted, or licensed datasets to a public repository. Keep local datasets out of Git using `.gitignore`.

## Running the experiments

Run commands from the repository root unless a script specifies otherwise.

### 1. CPST-MalariaNet forecasting

After placing `BdiDATASETS.xlsx` in the expected location:

```bash
python srcCode/data_loader.py
```

The script is configured to write experiment outputs under `Experiment1_Outputs/` (including figures and tables).

### 2. IoMT / HCPS security analysis

After placing `wustl-ehms-2020_with_attacks_categories.csv` in the expected location, examples include:

```bash
python srcCode/models.py
python srcCode/metrics.py
python srcCode/features.py
python srcCode/plots.py
python srcCode/pipeline.py
```

The scripts generate analysis results and/or plots. Check each script for its exact input columns, output filenames, and assumptions before running it.

## Reproducibility and interpretation

- Several scripts use fixed random seeds (for example, `random_state=42`) to support repeatable experiments.
- Results depend on dataset versions, preprocessing, train/test splits, package versions, and local configuration.
- The patient-outcome and resilience scenario calculations include modelling assumptions and illustrative scenario parameters. They should not be interpreted as validated clinical outcomes without independent validation.
- Inspect the scripts and verify their assumptions before using results in a publication or operational setting.

## Current limitations / recommended next steps

- Consolidate the two research strands into clearly separated folders if they are intended to remain in one repository.
- Add dataset provenance, versions, licences, and data dictionaries.
- Standardise file names and convert Python-in-`.txt` files to `.py`.
- Move hard-coded dataset paths into a configuration file or command-line arguments.
- Add tests for preprocessing, metric calculations, and scenario assumptions.
- Pin and test dependency versions in a clean environment.
- Document validated results only after reproducing the experiments.

## Citation

If you use this code in academic work, cite the associated paper or technical report once its bibliographic details are available. Please also cite the original datasets according to their licence and citation requirements.

## License

No license is currently specified. Unless a license is added, the code is not explicitly granted for reuse, redistribution, or modification. Add an appropriate license only after confirming that you have the right to license all included material.

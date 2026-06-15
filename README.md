# GC-MS Data Analysis

GC-MS fragmentation rule learning from 284k compounds.
Built with LightGBM + SHAP + spectral clustering + neutral loss analysis.

## Setup

```bash
uv pip install pandas numpy matplotlib scikit-learn lightgbm shap
uv pip install openpyxl pyarrow
```

## Data

data/MS_spectrum_data.xlsx
325k rows, 284k unique compounds
Columns: number, chemical_name, CAS, chemical_formular, MW, mz, intensity

## Pipeline

```bash
uv run python scripts/stage1_runner.py      # Clean + formula features
uv run python scripts/stage2_spectral.py    # Spectral binning
uv run python scripts/stage3_model.py       # Model + SHAP
uv run python scripts/improve_analysis.py   # CV + clustering + NL
uv run python scripts/visualize.py          # Charts
```

## Predict DBE

```bash
uv run python scripts/predict_dbe.py "15,26,27,39,51,52,77" "120,340,260,1110,400,320,999" 78.0
```

## Results

| Model | DBE R2 | MAE |
|-------|--------|-----|
| m/z only | 0.778 | 1.30 |
| NL only | 0.620 | 1.81 |
| m/z+NL | 0.811 | 1.20 |
| 5-fold CV | 0.783 | 1.33 |

## Key findings

- m/z 77 (C6H5+) appears in 81% of spectra, p(aromatic)=0.49
- m/z 91 (C7H7+) tropylium ion = phenyl derivative marker
- NL 15 (CH3 loss) strongest neutral loss predictor for DBE
- NL + m/z combined model outperforms either single view
- 50 spectral families automatically separate by chemical class
#!/bin/bash 
set -euo pipefail 
 
DATA=\"data/MS_spectrum_data.xlsx\" 
SCRIPT_DIR=\"scripts\" 
LOG_FILE=\"pipeline_$(date +%Y%m%d_%H%M%S).log\" 
 
echo \"=== GC-MS Pipeline ===\" 
date 
 
if [ ! -f \"$DATA\" ]; then 
    echo \"ERROR: Data not found: $DATA\" 
    exit 1 
fi 
 
mkdir -p output 
 
run_stage() { 
    echo \"\" 
    echo \"============================================================\" 
    echo \"[$(date +%H:%M:%S)] $2\" 
    echo \"============================================================\" 
    uv run python \"$SCRIPT_DIR/$1\" 2>&1 | tee -a \"$LOG_FILE\" 
    local ret=${PIPESTATUS[0]} 
    if [ $ret -ne 0 ]; then 
        echo \"FAILED: $1 (exit code $ret)\" 
        exit $ret 
    fi 
    echo \"[$(date +%H:%M:%S)] $2 - OK\" 
}
 
run_stage \"stage1_runner.py\" \"Stage 1: Data cleaning + formula features\" 
run_stage \"stage2_spectral.py\" \"Stage 2: Spectral vectorization\" 
run_stage \"stage3_model.py\" \"Stage 3: PCA + Clustering + LightGBM\" 
run_stage \"stage4_compare.py\" \"Stage 4: XGBoost comparison\" 
run_stage \"analysis_cv.py\" \"Stage 5: 5-fold CV + LOCO + m/z+NL\" 
run_stage \"shap_plots.py\" \"Stage 6: SHAP visualization\" 
run_stage \"visualize.py\" \"Stage 7: Report visualization\" 
 
echo \"\" 
echo \"Pipeline complete!\" 
date

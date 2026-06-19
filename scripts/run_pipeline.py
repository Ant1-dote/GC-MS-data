"""GC-MS pipeline: run all computation stages."""
import sys, time, subprocess
from pathlib import Path

SCRIPTS = Path("scripts")
DATA = Path("data") / "MS_spectrum_data.xlsx"

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] {label}")
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    ok = r.returncode == 0
    print(f"[{time.strftime('%H:%M:%S')}] -> {'OK' if ok else 'FAIL'} ({dt:.1f}s)")
    if not ok:
        sys.exit(r.returncode)
    return dt

if __name__ == "__main__":
    do_all = "--all" in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            start = int(a)
            break
    else:
        start = 1

    if not DATA.exists():
        print(f"ERROR: {DATA} not found")
        sys.exit(1)

    Path("output").mkdir(exist_ok=True)

    stages = [
        ("stage1_runner.py", "S1: Data+Feat"),
        ("stage2_spectral.py", "S2: Spectrum"),
        ("stage3_model.py", "S3: LGB+SHAP"),
        ("stage4_compare.py", "S4: XGBoost"),
        ("analysis_cv.py", "S5: CV+LOCO+NL"),
        ("shap_plots.py", "S6: SHAP"),
        ("visualize.py", "S7: Viz"),
    ]
    extras = [
        ("experiment_binning.py", "Exp: 0.5Da"),
        ("experiment_cnn.py", "Exp: CNN"),
    ]

    times = {}
    for i, (s, l) in enumerate(stages, 1):
        if i >= start:
            times[l] = run(["uv", "run", "python", str(SCRIPTS/s)], l)
    if do_all:
        for s, l in extras:
            times[l] = run(["uv", "run", "python", str(SCRIPTS/s)], l)

    print(f"\nTotal: {sum(times.values()):.1f}s")

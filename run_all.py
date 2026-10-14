"""
run_all.py  -  run the whole chain (scripts 01 to 08) for every CSV in a folder

Usage:   python run_all.py  <folder with the CSV files>

For every recording it runs 01 ... 07 (appending the results to results_all.csv), then
runs 08_evaluate.py once at the end. The two big temporary files (radar_cube.npy and
range_fft.npy, about 2 GB each for a 1730-frame recording) are deleted after every
recording unless KEEP_BIG_FILES is True.
"""

import os
import sys
import glob
import subprocess

KEEP_BIG_FILES = False

STEPS = [
    "01_csv_to_cube.py",
    "02_range_fft.py",
    "03_slow_time.py",
    "04_range_angle_map.py",
    "05_detect_targets.py",
    "06_mvdr_beamforming.py",
    "07_vital_signs.py",
]

folder = sys.argv[1] if len(sys.argv) > 1 else "."

files = sorted(glob.glob(os.path.join(folder, "*.csv")))

files = [f for f in files if os.path.basename(f) not in ("results_all.csv", "vital_results.csv")]

print(f"{len(files)} recording(s) found in {folder}")

for n, csv_file in enumerate(files, start=1):

    print("\n" + "=" * 70)
    print(f"[{n}/{len(files)}] {os.path.basename(csv_file)}")
    print("=" * 70)

    env = dict(os.environ, SIVIS_CSV=csv_file, SIVIS_NO_SHOW="1")

    try:
        for step in STEPS:
            print(f"\n--- {step} ---")
            subprocess.run([sys.executable, step], check=True, env=env)
    except subprocess.CalledProcessError:
        print(f"FAILED on {os.path.basename(csv_file)}: skipped.")

    if not KEEP_BIG_FILES:
        for tmp in ("radar_cube.npy", "range_fft.npy"):
            if os.path.exists(tmp):
                os.remove(tmp)

print("\n--- 08_evaluate.py ---")

subprocess.run([sys.executable, "08_evaluate.py"], check=False, env=dict(os.environ, SIVIS_NO_SHOW="1"))

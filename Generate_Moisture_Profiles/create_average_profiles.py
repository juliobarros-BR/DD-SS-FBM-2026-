#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 12:58:54 2025

@author: jortiz
"""

import pandas as pd
from pathlib import Path

# --- Config ---
base_folder = Path("./")  # directory containing all NEWwet_* folders
output_suffix = "_avg_moisture"


def folder_is_complete(original_folder: Path, mirror_folder: Path) -> bool:
    """Check if mirror folder exists and has same mask CSV files."""
    if not mirror_folder.exists():
        return False

    # Only mask CSV files
    orig_files = sorted([
        f.name for f in original_folder.glob("mask_*_cycle.csv")
    ])
    mirror_files = sorted([
        f.name for f in mirror_folder.glob("mask_*_cycle.csv")
    ])

    missing = sorted(set(orig_files) - set(mirror_files))
    if missing:
        print(f"⚙️  Incomplete: missing {len(missing)} file(s) in {mirror_folder.name}")
        return False

    return True


def is_binary(path):
    """Detect binary files to avoid UnicodeDecodeError."""
    with open(path, "rb") as fh:
        start = fh.read(16)
        return (b"\x00" in start or
                start.startswith(b"\x89PNG") or
                start.startswith(b"\x1f\x8b"))  # PNG/GZIP signatures


for subfolder in sorted(base_folder.iterdir()):
    # print(subfolder)
    if not subfolder.is_dir() or not subfolder.name.startswith("Moisture_Profiles_"):
        continue

    if subfolder.name.endswith(output_suffix):
        print(f"⏩ Skipping {subfolder.name} (already averaged).")
        continue

    print(f"\n🔍 Checking folder: {subfolder.name}")

    # Find mask drying/moistening CSVs only
    files = sorted(subfolder.glob("mask_*_cycle.csv"))
    if not files:
        print("⚠️  No mask_*_cycle.csv files found — skipping.")
        continue

    mirror_folder = subfolder.parent / f"{subfolder.name}{output_suffix}"

    # If all files already processed → skip
    if folder_is_complete(subfolder, mirror_folder):
        print(f"✓ Skipping {subfolder.name} — already complete.")
        continue

    mirror_folder.mkdir(exist_ok=True)
    print(f"⚙️  Processing {subfolder.name} (creating/updating {mirror_folder.name})")

    for f in files:
        out_file = mirror_folder / f.name

        # Skip if already written and non-empty
        if out_file.exists() and out_file.stat().st_size > 100:
            continue

        # Skip binary files
        if is_binary(f):
            print(f"   ⚠️ Skipping {f.name}: binary file detected")
            continue

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"   ⚠️ Could not read {f.name}: {e}")
            continue

        # Compute average moisture per time step
        avg = df.groupby("time_index")["moisture"].mean().reset_index()
        avg = avg.rename(columns={"moisture": "avg_moisture"})

        # Merge this average back to all fibers
        df2 = df.merge(avg, on="time_index", how="left")

        # Replace per-fiber moisture with bundle average
        df2["moisture"] = df2["avg_moisture"]
        df2 = df2.drop(columns=["avg_moisture"])

        df2.to_csv(out_file, index=False)
        print(f"   → Saved {out_file.name} ({len(df2)} rows)")

print("\n✅ All folders checked and updated.")

#!/usr/bin/env python3
"""
Post-hoc reproduction helper for Step312 p-value reporting.

IMPORTANT:
This is a publication/reproducibility helper written after the original
Step312 execution. It is NOT the original script used to generate the
2026-11-16 Step312 output.

It reads the consolidated Step311 CSV, recomputes the add-one Monte Carlo
tail estimate using each row's valid_bootstrap count, and writes a new
reproduction CSV and manifest. It never overwrites the locked source CSV
or the original Step312 output.
"""

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "reproducibility" / "results"
MANIFESTS = ROOT / "docs" / "reproducibility" / "manifests"

INPUT = RESULTS / "step237_final_regime_bootstrap_corrected_locked.csv"
OUTPUT = RESULTS / "step237_final_regime_bootstrap_corrected_locked_pvalues_reproduced.csv"
MANIFEST = MANIFESTS / "step312_reproduction_manifest.json"

METHOD = (
    "Two-sided empirical bootstrap tail estimate with add-one correction "
    "using the row-specific valid replicate count"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Required Step311 input not found: {INPUT}")

    if INPUT.resolve() == OUTPUT.resolve():
        raise RuntimeError("Input and output paths must be different.")

    with INPUT.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")

        fieldnames = list(reader.fieldnames)
        required = {"p_two_sided", "valid_bootstrap"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if "p_two_sided_plus_one" not in fieldnames:
            fieldnames.append("p_two_sided_plus_one")
        if "p_value_method" not in fieldnames:
            fieldnames.append("p_value_method")

        rows = list(reader)

    if len(rows) != 48:
        raise ValueError(f"Expected 48 regime-metric rows; found {len(rows)}.")

    replicate_counts = []

    for row in rows:
        p = float(row["p_two_sided"])
        r = int(row["valid_bootstrap"])

        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Invalid empirical p-value: {row}")
        if r <= 0:
            raise ValueError(f"Invalid valid_bootstrap count: {row}")

        corrected = min(1.0, (p * r + 2.0) / (r + 1.0))
        row["p_two_sided_plus_one"] = repr(corrected)
        row["p_value_method"] = METHOD
        replicate_counts.append(r)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "experiment": "Step312_RowSpecific_PValue_Reproduction",
        "script_type": "post-hoc reproduction helper; not original Step312 source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "output_file": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "rows": len(rows),
        "valid_bootstrap_min": min(replicate_counts),
        "valid_bootstrap_max": max(replicate_counts),
        "replicate_denominator": "row-specific valid_bootstrap",
        "p_value_formula": "min(1, (p_two_sided * valid_bootstrap + 2) / (valid_bootstrap + 1))",
        "confidence_intervals_changed": False,
        "point_estimates_changed": False,
        "adaptation_gate_changed": False,
        "note": (
            "This helper reproduces the documented Step312 calculation "
            "from the consolidated input. It does not replace the original "
            "execution record or overwrite the original Step312 output."
        ),
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Rows reproduced: {len(rows)}")
    print(f"Valid bootstrap range: {min(replicate_counts)} to {max(replicate_counts)}")
    print(f"Output: {OUTPUT}")
    print(f"Output SHA256: {manifest['output_sha256']}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
run_main_results.py — Legacy experiment launcher (stub).

The full main-result tables in the paper were produced by batch jobs on the
experiment VM and are released here as `results/master/*/run_summary.csv`
and `results/rerun/*/run_summary.csv`.

The paper tables are regenerated from these CSV files using:

    python3 paper_figures/compute_paper_tables.py

This file was an early development launcher and is retained only to preserve
git history. It is not invoked by the released artifact.
"""
import sys

MESSAGE = """\
This is a legacy launcher and is no longer functional.

The released artifact reproduces all paper tables from the included
run_summary.csv files. To regenerate the tables, run:

    python3 paper_figures/compute_paper_tables.py

See the 'Note on experiment launchers' section in README.md for details.
"""

if __name__ == "__main__":
    sys.stderr.write(MESSAGE)
    sys.exit(1)

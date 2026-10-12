"""Project paths (single definition).

CODE    : this repository
ROOT    : large raw and intermediate data (SWORD, SWOT, gauges, MCMC output), kept on local disk.
          Override with the environment variable P38_DATA.
TABLES  : final tables (output/)
FIGURES : paper figures (figures/)
"""
import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("P38_DATA", Path.home() / "swot_p38_work")).expanduser()
TABLES = CODE / "output"
FIGURES = CODE / "figures"
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

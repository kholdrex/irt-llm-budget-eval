"""Working-directory layout. Override the root with the IRTEVAL_WORKDIR environment variable."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("IRTEVAL_WORKDIR", Path.cwd())).resolve()
DATA = WORKDIR / "data"
RESULTS = WORKDIR / "results"
TABLES = RESULTS / "tables"
FIGURES = WORKDIR / "figures"


def ensure_dirs():
    for d in (DATA, RESULTS, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)

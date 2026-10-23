"""
toposnail.__main__: CLI entrypoint for TopoSnail topology optimization workstation.
Enables execution via `python -m toposnail [options]`.
"""

from __future__ import annotations

import sys
from toposnail.cad_to_bionic import main


if __name__ == "__main__":
    sys.exit(main())

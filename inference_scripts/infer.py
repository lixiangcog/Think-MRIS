"""Canonical Think-MRIS inference entry point.

The implementation supports one or more grounded targets, so both public entry
points intentionally share the paper-calibrated VLKI + SAM2 pipeline.
"""

from infer_multi_object import main


if __name__ == "__main__":
    main()

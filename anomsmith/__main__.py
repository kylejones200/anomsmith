"""Entry point for `python -m anomsmith` and the `dca` console script."""

import sys


def main() -> int:
    """Run anomsmith CLI.

    Currently provides a minimal entry point. Extend for future CLI features.
    """
    print("anomsmith: Anomaly detection workflows", file=sys.stderr)
    print(
        "Usage: Use as a library - from anomsmith import detect_anomalies, ...",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

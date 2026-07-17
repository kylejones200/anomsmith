"""Pytest configuration for the anomsmith test suite.

Continuous integration runs this suite on **Linux only** (Ubuntu 24.04 in
``.github/workflows/ci.yml`` and ``release.yml``). There is no macOS or
Windows job matrix; local runs on other platforms are best-effort.
"""

from __future__ import annotations

from typing import Any


def pytest_report_header(config: Any) -> list[str]:  # noqa: ARG001 — name fixed by pytest hookspec
    return [
        "anomsmith: CI tests run on Linux (ubuntu-24.04); see .github/workflows/ci.yml",
    ]

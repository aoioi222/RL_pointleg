"""Collect the source and standalone simulation regression suites."""

from pathlib import Path
import unittest


def load_tests(loader, tests, pattern):
    root = Path(__file__).resolve().parents[1]
    suite = unittest.TestSuite()
    for directory in (root / "src" / "pointleg" / "tests", root / "sim2sim" / "tests"):
        suite.addTests(unittest.TestLoader().discover(str(directory), pattern=pattern or "test*.py"))
    return suite

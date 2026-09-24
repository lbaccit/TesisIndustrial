"""
run_tests.py

Runs the whole test suite with unittest, no external dependencies needed.

Usage:
    python3 run_tests.py            # normal
    python3 run_tests.py -q         # one line per file only

Equivalent to:
    python3 -m unittest discover -p "test_*.py" -v

Exit code is 0 when everything passes and 1 otherwise, so it can be wired into
CI later on.

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import os
import sys
import unittest

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)

    verbosity = 1 if "-q" in sys.argv else 2

    suite = unittest.defaultTestLoader.discover(here, pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print()
    print("=" * 70)
    print(f"Tests run: {result.testsRun}   "
          f"failures: {len(result.failures)}   errors: {len(result.errors)}")
    print("RESULT:", "ALL TESTS PASSED" if result.wasSuccessful() else "FAILED")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)

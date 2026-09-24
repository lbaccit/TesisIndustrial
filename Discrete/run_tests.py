"""
run_tests.py

Runs the whole test suite with unittest, no external dependencies needed.

Usage:
    python3 run_tests.py            # normal
    python3 run_tests.py -q         # one line per file only

Equivalent to:
    python3 -m unittest discover -p "test_*.py" -v

Exit code is 0 when everything passes and 1 otherwise, so it can be wired into
CI later on. This is the counterpart of jMarkov's ``test/jphase/AllTests.java``.

Correspondence with jMarkov's jphase tests
------------------------------------------
    Java (continuous only)          Python (discrete)
    -----------------------------   ------------------------------
    DenseContProbTest.java          test_DenseDiscProb.py
    DenseContMomentTest.java        test_DenseDiscMoment.py
    DenseContClosureTest.java       test_DenseDiscClosure.py
    SparseContProbTest.java         test_SparseDiscProb.py
    SparseContMomentTest.java       test_SparseDiscMoment.py
    SparseContClosureTest.java      test_SparseDiscClosure.py
    (none)                          test_AbstractDiscPhaseVar.py
    (none)                          test_DenseDiscPhaseVar.py
    (none)                          test_SparseDiscPhaseVar.py
    (none)                          test_matrix_utils.py

The fixtures are in ``discrete_fixtures.py`` and the expected values in
``reference_values.py``, which ``generate_reference_values.py`` regenerates.

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

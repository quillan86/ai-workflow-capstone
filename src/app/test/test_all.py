import unittest
import sys


def run_all_tests() -> unittest.TestResult:
    loader = unittest.TestLoader()
    start_dir = "./"
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    res: unittest.TestResult = runner.run(suite)
    return res

if __name__ == '__main__':

    res = run_all_tests()
    sys.exit(0 if res.wasSuccessful() else 1)

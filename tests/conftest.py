import os
import sys

# let tests import the rag package and tests/fakes.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

SAMPLE_PDF = os.path.join(ROOT, "eval", "sample_manual.pdf")

from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH.parent / 'skuld-lib'))
from heat_runner import run

if __name__ == '__main__':
    run(2, __file__)

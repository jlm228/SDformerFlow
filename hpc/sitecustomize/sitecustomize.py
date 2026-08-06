"""Auto-imported by Python's `site` module at interpreter startup, for every process that
sources hpc/env.sh (which puts this directory on PYTHONPATH). Not imported explicitly anywhere
in this codebase -- that is how sitecustomize.py works.

Purpose: patch third-party library incompatibilities with the numpy/torch/mlflow versions this
project's venv actually has installed, without editing site-packages (pip would silently revert
any such edit on the next install) and without having to remember to patch every entry script
that happens to trigger the problem.
"""

import numpy as np

# spikingjelly==0.0.0.0.14's CUDA kernel dtype dispatcher still does `value.dtype == np.int` at
# activation_based/auto_cuda/base.py:249. `np.int` was removed in NumPy 1.24+ (deprecated since
# 1.20) -- see https://github.com/fangwei123456/spikingjelly/issues/583, open and unfixed as of
# this pin. NumPy's own deprecation message says restoring the alias as plain `int` "will not
# modify any behavior and is safe", which is exactly what this does; `np.dtype(int)` resolves to
# the platform's default int type, matching np.int's pre-deprecation meaning.
if not hasattr(np, "int"):
    np.int = int

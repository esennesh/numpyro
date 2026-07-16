# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

try:
    import funsor  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Looking like you want to use QEM, which builds on the massively parallel "
        "importance weighting (MPIW) machinery, an experimental feature in NumPyro. "
        "You need to install `funsor` to be able to use this module. It can be "
        "installed with `pip install funsor`."
    ) from e

from numpyro.contrib.qem.core import QEM, QEMRunResult, QEMState

__all__ = [
    "QEM",
    "QEMRunResult",
    "QEMState",
]

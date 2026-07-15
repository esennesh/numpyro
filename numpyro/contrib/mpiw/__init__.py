# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

try:
    import funsor  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Looking like you want to use the massively parallel importance weighting "
        "(MPIW) machinery, which is an experimental feature in NumPyro. You need to "
        "install `funsor` to be able to use this module. It can be installed with "
        "`pip install funsor`."
    ) from e

from numpyro.contrib.mpiw.contraction import (
    NamedFactor,
    contract_log_marginal,
    contract_with_source_terms,
)
from numpyro.contrib.mpiw.core import MPIW

__all__ = [
    "MPIW",
    "NamedFactor",
    "contract_log_marginal",
    "contract_with_source_terms",
]

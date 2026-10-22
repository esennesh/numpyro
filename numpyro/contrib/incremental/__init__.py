# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Incremental density computation (EXPERIMENTAL).

Compiles a NumPyro model to a deterministic density program (a jaxpr whose inputs are
the latent site values and whose outputs are the per-factor unsummed log densities) and
incrementalizes that program against typed change descriptions, following Zaiser et
al., *Incremental Computation for Efficient Programmable Inference in Probabilistic
Programs* (PLDI 2026, doi:10.1145/3808316).
"""

from numpyro.contrib.incremental.delta import Change, Dense, Sliced
from numpyro.contrib.incremental.density import (
    DensityProgram,
    FactorMeta,
    SiteMeta,
    trace_density,
)
from numpyro.contrib.incremental.incremental import (
    IncrementalDensity,
    IncrementalState,
)
from numpyro.contrib.incremental.interpreter import incrementalize

__all__ = [
    "Change",
    "Dense",
    "DensityProgram",
    "FactorMeta",
    "IncrementalDensity",
    "IncrementalState",
    "SiteMeta",
    "Sliced",
    "incrementalize",
    "trace_density",
]

# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Find, Weigh, Learn: fast MAP estimation in graphical models.

An implementation of the procedure described in *Find, Weigh, Learn: Fast MAP
Estimation in Graphical Models*. :func:`~numpyro.contrib.fwl.find_weigh_learn`
locates joint modes of a model's posterior, wraps a mixture-of-Gaussians
importance-sampling proposal around them, and returns that proposal as a NumPyro
guide together with differentiable bounds on ``log Z(theta)``.

Requires :mod:`optimistix` for the continuous inner optimization, and
:mod:`funsor` when the model has discrete latent sites.
"""

from numpyro.contrib.fwl.api import FWLResult, find_weigh_learn
from numpyro.contrib.fwl.junction import CliqueTree, build_clique_tree
from numpyro.contrib.fwl.options import FWLOptions
from numpyro.contrib.fwl.structure import LatentPacking, ModelStructure, analyze

__all__ = [
    "CliqueTree",
    "FWLOptions",
    "FWLResult",
    "LatentPacking",
    "ModelStructure",
    "analyze",
    "build_clique_tree",
    "find_weigh_learn",
]

# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
from zoo import ZOO


@pytest.fixture(params=sorted(ZOO), ids=sorted(ZOO))
def zoo_entry(request):
    return request.param, ZOO[request.param]

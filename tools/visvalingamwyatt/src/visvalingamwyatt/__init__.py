# -*- coding: utf-8 -*-
# This file is part of visvalingamwyatt.
# https://github.com/fitnr/visvalingamwyatt
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, fitnr <contact@fakeisthenewreal.org>
'''
Visvalingam-Whyatt method of poly-line vertex reduction

Visvalingam, M and Whyatt J D (1993)
"Line Generalisation by Repeated Elimination of Points", Cartographic J., 30 (1), 46 - 51

Described here:
http://web.archive.org/web/20100428020453/http://www2.dcs.hull.ac.uk/CISRG/publications/DPs/DP10/DP10.html

source: https://github.com/Permafacture/Py-Visvalingam-Whyatt/
'''
from . import visvalingamwyatt
from .visvalingamwyatt import Simplifier, simplify, simplify_feature, simplify_geometry

__version__ = '0.3.0'

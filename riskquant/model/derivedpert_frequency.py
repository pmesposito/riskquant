"""A PERT model suitable for frequency. Returns an array of ints.
"""

#   Copyright 2019-2020 Netflix, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import tensorflow_probability as tfp


class DerivedPERTFrequency(object):

    def __init__(self, min_tef, max_tef, most_likely_tef, min_vuln, max_vuln, most_likely_vuln, kurtosis_tef, kurtosis_vuln):
        """:param frequency = Mean rate per interval"""
        if min_tef >= max_tef or min_vuln >= max_vuln:
            # Min frequency must exceed max frequency
            raise AssertionError
        if (not min_tef <= most_likely_tef <= max_tef) or (not min_vuln <= most_likely_vuln <= max_vuln):
            # Most likely should be between min and max frequencies.
            raise AssertionError

        # Set up the PERT distribution
        # From FAIR: the most likely frequency will set the skew/peak, and
        # the "confidence" in the most likely frequency will set the kurtosis/temp of the distribution.
        self.distribution_tef = tfp.experimental.substrates.numpy.distributions.PERT(
            low=min_tef, peak=most_likely_tef, high=max_tef, temperature=kurtosis_tef)
        self.distribution_vuln = tfp.experimental.substrates.numpy.distributions.PERT(
            low=min_vuln, peak=most_likely_vuln, high=max_vuln, temperature=kurtosis_vuln)

    def draw(self, n=1):
        return [np.random.poisson(x) for x in self.distribution_tef.sample(n) * self.distribution_vuln.sample(n)]

    def mean(self):
        return self.distribution.mode().flat[0]

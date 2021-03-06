from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Bernoulli(TorchDistribution):
    """
    Bernoulli distribution.

    Distribution over a vector of independent Bernoulli variables. Each element
    of the vector takes on a value in `{0, 1}`.

    This is often used in conjunction with `torch.nn.Sigmoid` to ensure the
    `ps` parameters are in the interval `[0, 1]`.

    :param torch.autograd.Variable ps: Probabilities. Should lie in the
        interval `[0,1]`.
    :param logits: Log odds, i.e. :math:`\\log(\\frac{p}{1 - p})`. Either `ps` or
        `logits` should be specified, but not both.
    :param batch_size: The number of elements in the batch used to generate
        a sample. The batch dimension will be the leftmost dimension in the
        generated sample.
    :param log_pdf_mask: Tensor that is applied to the batch log pdf values
        as a multiplier. The most common use case is supplying a boolean
        tensor mask to mask out certain batch sites in the log pdf computation.
    """

    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.Bernoulli(probs=ps, logits=logits)
        x_shape = ps.size() if ps is not None else logits.size()
        event_dim = 1
        super(Bernoulli, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

    def enumerate_support(self):
        """
        Returns the Bernoulli distribution's support, as a tensor along the first dimension.

        Note that this returns support values of all the batched RVs in lock-step, rather
        than the full cartesian product. To iterate over the cartesian product, you must
        construct univariate Bernoullis and use itertools.product() over all univariate
        variables (may be expensive).

        :return: torch variable enumerating the support of the Bernoulli distribution.
            Each item in the return value, when enumerated along the first dimensions, yields a
            value from the distribution's support which has the same dimension as would be returned by
            sample.
        :rtype: torch.autograd.Variable.
        """
        return super(Bernoulli, self).enumerate_support()

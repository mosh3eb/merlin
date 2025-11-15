# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Specialized loss functions for QML
"""

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class NKernelAlignment(_Loss):
    r"""
    Negative kernel-target alignment loss function for quantum kernel training.

    Within quantum kernel alignment, the goal is to maximize the
    alignment between the quantum kernel matrix and the ideal
    target matrix given by :math:`K^{*} = y y^T`, where
    :math:`y \in \{-1, +1\}` are the target labels.

    The negative kernel alignment loss is given as:

    .. math::

        \text{NKA}(K, K^{*}) =
        -\frac{\operatorname{Tr}(K K^{*})}{
        \sqrt{\operatorname{Tr}(K^2)\operatorname{Tr}(K^{*2})}}
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.dim() != 2:
            raise ValueError(
                "Input must be a 2D tensor representing the kernel matrix."
            )

        if torch.any((target != 1) & (target != -1)):
            raise ValueError(
                "Negative kernel alignment requires binary target values +1, -1."
            )

        if target.dim() == 1:
            # Make the target the ideal Kernel matrix
            target = target.unsqueeze(1) @ target.unsqueeze(0)

        numerator = torch.sum(input * target)
        denominator = torch.linalg.norm(input) * torch.linalg.norm(target)
        return -numerator / denominator

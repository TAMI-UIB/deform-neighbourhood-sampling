import logging
import unittest

import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck

import deform_neighbourhood_sampling


class TestDeformNeighbourhood(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):

        nbrhd_h = 3
        nbrhd_w = 3
        stride_h = 1
        stride_w = 1
        pad_h = 1
        pad_w = 1
        dilation_h = 1
        dilation_w = 1
        offset_groups = 1

        options_1 = [nbrhd_h, nbrhd_w, stride_h, stride_w,
                     pad_h, pad_w, dilation_h, dilation_w, offset_groups]
        options_2 = [1, 1,
                     1, 1,
                     0, 0,
                     1, 1,
                     2]

        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        return [
            [make_tensor(1, 1, 20, 20), make_tensor(
                1, 18, 20, 20), *options_1],
            [make_tensor(4, 3, 20, 20), make_tensor(
                4, 18, 20, 20), *options_1],
            [make_tensor(1, 2, 30, 30), make_tensor(
                1, 4, 30, 30), *options_2],
            [make_tensor(4, 6, 30, 30), make_tensor(
                4, 4, 30, 30), *options_2],
        ]

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            opcheck(
                torch.ops.deform_neighbourhood_sampling.deform_neighbourhood.default, args)

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


if __name__ == "__main__":
    unittest.main()

#! /usr/bin/env
# -*- encoding:utf-8 -*-

import unittest
import torch
from lempa import sum_product
from lempa import spmat
from torch.autograd import Variable


class TestDecoderModel(unittest.TestCase):

    def test_sum_product_decoding(self):
        filename = 'data/3x6irRegLDPC/parity_check_matrix.spmat'
        pcm = spmat.read_spmat(filename)
        codedir = 'data/3x6irRegLDPC'
        code = sum_product.Code(parity_check_matrix=pcm)
        num_of_iteration = 3
        model = sum_product.NeuralSumProductModel(
            code, num_of_iteration,
            variable_node_normalization=False,
            check_node_normalization=False)
        y = torch.Tensor([
            [1.620803, 0.264281, -0.031637, -0.127654, 0.746347, 1.003543]
            for _ in range(5)])
        var = 0.794328
        llr = Variable(2 * y / var)
        output = model(llr)[-1].data

        expected = torch.Tensor([
            [4.3974, 1.6925, 1.7111, 1.7111, 2.0840, 2.7033]
            for _ in range(5)
        ])

        eps = 1e-3
        self.assertTrue((torch.abs(output - expected) < eps).all())


if __name__ == '__main__':
    unittest.main()

#! /usr/bin/env
# -*- encoding:utf-8 -*-

import unittest
import numpy as np
# import neural_sum_product_decoder
# from neural_sum_product_decoder import spmat
# import lempa
from lempa import spmat


class TestSpmat(unittest.TestCase):

    def test_is_rref(self):
        test_cases = [
            (np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]]), True),
            (np.array([
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0]]), True),
            (np.array([
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0]]), False),
            (np.array([
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 1, 0]]), False),
            (np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 1]]), False)
        ]

        for arg, expected in test_cases:
            self.assertEqual(spmat.is_rref(arg), expected)

    def test_transform_to_rref(self):
        # random test
        for width in [10, 100]:
            for height in [10, 100]:
                success_num = 0
                trial_num = 10
                for _ in range(trial_num):
                    m = np.random.randint(0, 2, [height, width])
                    spmat.transform_to_rref(m)
                    if spmat.is_rref(m):
                        success_num += 1
                self.assertEqual(success_num, trial_num)

    def test_read_spmat(self):
        hamming_code = np.array([
            [1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ])

        with open('data/hamming_code/parity_check_matrix.spmat') as f:
            m = spmat.read_spmat(f)
        m = m.to_ndarray()
        self.assertEqual(np.all(m == hamming_code), True)

    def test_transform_to_generator_matrix(self):
        for n in [10, 100, 1000]:
            for m in [10, 100, 1000]:
                parity_check_matrix = np.random.randint(0, 2, [m, n])
                generator_matrix = spmat.to_generator_matrix(
                    parity_check_matrix)
                if generator_matrix.shape[0] != 0:
                    self.assertEqual(
                        spmat.verify_generator_matrix(
                            generator_matrix,
                            parity_check_matrix),
                        True)

    def test_to_gather_matrix(self):
        with open('data/hamming_code/parity_check_matrix.spmat') as f:
            m = spmat.read_spmat(f)
        gather_matrix = m.to_gather_matrix().to_ndarray()

        expected = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]])

        self.assertEqual(np.all(gather_matrix == expected), True)

    def test_variable_node_matrix(self):
        with open('data/hamming_code/parity_check_matrix.spmat') as f:
            m = spmat.read_spmat(f)
        vmatrix = m.to_variable_node_matrix().to_ndarray()

        expected = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.assertEqual(np.all(vmatrix == expected), True)

    def test_check_node_matrix(self):
        with open('data/hamming_code/parity_check_matrix.spmat') as f:
            m = spmat.read_spmat(f)
        cmatrix = m.to_check_node_matrix().to_ndarray()

        expected = np.array([
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])

        self.assertEqual(np.all(cmatrix == expected), True)


if __name__ == '__main__':
    unittest.main()

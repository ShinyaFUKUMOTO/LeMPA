# -*- encoding:utf-8 -*-

import numpy as np


def find(iterable, element, errval=None):
    r'''
    Return the first index of the element.

    Args:
        iterable: an iterable object
        element: an element to search for
        errval: a return value when no matchin the element is founded
            Default: None

    Returns:
        The first index of the element. If no matching the element is founded,
        return the errval (default: None).

    Example:
        >>> find([1,2,3], 2)
        1
        >>> find([1,2,3], 4)
        None
        >>> find([1,2,3], 4, -1)
        -1
    '''
    for i, e in enumerate(iterable):
        if e == element:
            return i

    return errval


def make_spmat_from_ndarray(ndarray):
    r'''Make the binary matrix (Spmat) from the matrix (numpy.ndarray)


        Note:
            The generated binary matrix has 1 where the given matrix has
            non-zero element.

        Args:
            ndarray: numpy ndarray object

        Returns:
            Spmat object

        Example:
            >>> import numpy as np
            >>> ndarray = np.array([[1,0], [0, 2])
            >>> make_spmat_from_ndarray(ndarray)
            [[1, 0],
             [0, 1]]
    '''
    shape = ndarray.shape
    m = Spmat(shape)
    for ri in range(shape[0]):
        for ci in range(shape[1]):
            if ndarray[ri, ci] != 0:
                m.entry(ri, ci)
    return m


def transform_to_rref(matrix):
    r'''Transform a given binary matrix to reduced row echelon form.

    Args:
        matrix (numpy.ndarray): the binary matrix to transform

    Returns:
        The rank of the matrix.

    Example:
        >>> import numpy as np
        >>> a = np.array([[1, 0, 1, 1],
                          [0, 0, 1, 1],
                          [1, 1, 1, 0]])
        >>> transform_to_rref(a)
        3
        >>> a
        array([[1, 0, 0, 0],
               [0, 1, 0, 1],
               [0, 0, 1, 1]])
        >>>
        >>> b = np.array([[1, 1, 1, 1],
                          [0, 0, 1, 1],
                          [1, 1, 1, 0]])
        >>> transform_to_rref(b)
        3
        >>> b
        array([[1, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
        >>>
        >>> c = np.array([[1, 1, 1, 1],
                          [0, 0, 1, 1],
                          [1, 1, 0, 0]])
        >>> transform_to_rref(c)
        2
        >>> c
        array([[1, 1, 0, 0],
               [0, 0, 1, 1],
               [0, 0, 0, 0]])
    '''
    shape = matrix.shape
    ri = 0  # matrix[:ri, :] has already been transformed region
    for ci in range(shape[1]):
        # Find a row such that the index of leading coefficient equalt to ci
        for rj in range(ri, shape[0]):
            row = matrix[rj, :]
            are_zeros = 1 not in row[:ci]
            if are_zeros and row[ci] != 0:
                break
        else:
            continue

        # Erase non zero elements in ci-th column
        for rk in range(shape[0]):
            if rj == rk:
                continue
            if matrix[rk, ci] != 0:
                matrix[rk, :] ^= matrix[rj, :]

        # Swap the positions of two rows to keep transformed region
        if ri != rj:
            matrix[[ri, rj]] = matrix[[rj, ri]]

        ri += 1

    return ri


def to_rref(matrix):
    r'''Out-of-place version of transform_to_rref

    Args:
        matrix: the binary matrix to transform

    Returns:
        matrix (ndarray): a reduced row echelon form of the given matrix
        rank (int): the rank of the matrix.

    Example:
        >>> import numpy as np
        >>> a = np.array([[1, 0, 1, 1],
                          [0, 0, 1, 1],
                          [1, 1, 1, 0]])
        >>> rank, mat = to_rref(a)
        >>> rank
        3
        >>> mat
        array([[1, 0, 0, 0],
               [0, 1, 0, 1],
               [0, 0, 1, 1]])
    '''
    rref_matrix = matrix.copy()
    rank = transform_to_rref(rref_matrix)
    return rref_matrix, rank


def is_rref(matrix):
    '''Find whether the given binary matrix is reduced row echelon form'''
    i = -1
    for row in matrix:
        j = find(row, 1)

        if i is None:
            if j is None:
                continue
            else:
                return False

        if j is None:
            continue

        if j <= i:
            return False

        if np.count_nonzero(matrix[:, j]) > 1:
            return False
        i = j

    return True


def to_generator_matrix(parity_check_matrix):
    r'''Return a generator matrix corresponding to a given binary parity check
    matrix.

    Args:
        parity_check_matrix (numpy.ndarray): a parity check matrix of the code

    Returns:
        a generator matrix of the code

    Example:
        >>> import numpy as np
        >>> pcm = np.array([
            [1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ])
        >>> gm = to_generator_matrix(pcm)
        >>> gm
        array([[1, 1, 1, 0, 0, 0, 0],
               [0, 1, 0, 1, 1, 0, 0],
               [1, 1, 0, 1, 0, 1, 0],
               [1, 0, 0, 1, 0, 0, 1]])
        >>> # product of gm and pcm must be zero matrix
        >>> (np.matmul(gm, pcm.T) % 2 == 0).all()
        True
    '''
    matrix, rank = to_rref(parity_check_matrix)
    if rank == 0:
        return np.eye(0, dtype=int)
    matrix = matrix[:rank]
    _, n = matrix.shape

    information_bits_indices = []
    i = 0
    for row in matrix:
        i += find(row[i:], 1)
        information_bits_indices.append(i)

    diff = set(range(n)).difference(information_bits_indices)
    parity_bits_indices = list(diff)

    # Permutate & Transpose (H = [A I] -> G = [I A^T])
    systematic_order = parity_bits_indices + information_bits_indices

    k = n - rank
    if len(parity_bits_indices) != 0:
        matrix = np.concatenate(
            (np.eye(k, dtype=int), matrix[:, parity_bits_indices].T),
            axis=1
        )
    else:
        matrix = np.eye(k, dtype=int)

    # Inverse permutation
    matrix[:, systematic_order] = matrix[:, range(n)]

    return matrix


def verify_generator_matrix(generator_matrix,
                            parity_check_matrix):
    '''
        Verify whether a given generator matrix G correspond correctly to a
        given parity check matrix H
    '''
    return 1 not in np.matmul(parity_check_matrix, generator_matrix.T) % 2


class Spmat:
    r'''This is a class for sparse binary matrix'''

    def __init__(self, shape, array=None):
        self._shape = shape
        self._row_list = [[] for ri in range(shape[0])]
        self._col_list = [[] for ci in range(shape[1])]

        if isinstance(array, np.ndarray):
            _construct_from_ndarray(array)

    def _construct_from_ndarray(self, ndarray):
        shape = ndarray.shape
        for ri in range(shape[0]):
            for ci in range(shape[1]):
                if ndarray[ri, ci] != 0:
                    self.entry(ri, ci)

    def nonzero(self):
        r'''Return list of indices of non-zero elements in row major

        Example:
            >>> Spmat(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])).nonzero()
            [[0, 2], [1], [0, 2]]
        '''

        return [list(row_indices) for row_indices in self._row_list]

    def to_ndarray(self):
        r'''Return the self matrix as numpy.ndarray object

        Returns:
            matrix (numpy.ndarray): the binary matrix
        '''
        m = np.zeros(self._shape, dtype=int)
        for ri, col_indices in enumerate(self._row_list):
            for ci in col_indices:
                m[ri, ci] = 1

        return m

    def entry(self, i, j):
        r'''Entry non-zero element at the i-th row and j-th column'''
        self._row_list[i].append(j)
        self._col_list[j].append(i)

    def copy(self):
        r'''Returns a copy of the self matrix'''
        mat = Spmat(self._shape)
        for ri, col_indices in enumerate(self._row_list):
            for ci in col_indices:
                mat.entry(ri, ci)

        return mat

    def transpose(self):
        r'''Returns a transpose of the self matrix'''
        mat = self.copy()
        mat._shape = mat._shape[::-1]
        mat._row_list, mat._col_list = mat._col_list, mat._row_list

        return mat

    def count_nonzero(self):
        r'''Counts the number of non-zero elements'''
        return sum([len(col_indices) for col_indices in self._row_list])

    def to_gather_matrix(self):
        r'''Returns a gather matrix of the self parity check matrix

            see also SumProductAlgorithm class

            Example:
            >>> import numpy as np
            >>> pcm = np.array([[1, 0, 1, 1, 1, 0, 0],
                                [1, 1, 0, 1, 0, 1, 0],
                                [0, 1, 1, 1, 0, 0, 1]])
            >>> spmat = make_spmat_from_ndarray(pcm)
            >>> spmat.to_gather_matrix().to_ndarray()
            array([[1, 0, 0, 0, 0, 0, 0],
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
            '''
        shape = (self.count_nonzero(), self._shape[1])
        gather_matrix = Spmat(shape)

        ri = 0
        for col_indices in self._row_list:
            for ci in col_indices:
                gather_matrix.entry(ri, ci)
                ri += 1

        return gather_matrix

    def to_scatter_matrix(self):
        r'''Returns a scatter matrix of the self parity check matrix

        see also SumProductAlgorithm class

        Exapmle:
            >>> import numpy as np
            >>> pcm = np.array([[1, 0, 1, 1, 1, 0, 0],
                                [1, 1, 0, 1, 0, 1, 0],
                                [0, 1, 1, 1, 0, 0, 1]])
            >>> spmat = make_spmat_from_ndarray(pcm)
            >>> spmat.to_scatter_matrix().to_ndarray()
            array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        '''
        return self.to_gather_matrix().transpose()

    def to_check_node_matrix(self):
        r'''Returns a check node matrix of the self parity check matrix

        see SumProductAlgorithm class

        Example:
            >>> import numpy as np
            >>> pcm = np.array([[1, 0, 1, 1, 1, 0, 0],
                                [1, 1, 0, 1, 0, 1, 0],
                                [0, 1, 1, 1, 0, 0, 1]])
            >>> spmat = make_spmat_from_ndarray(pcm)
            >>> spmat.to_check_node_matrix().to_ndarray()
            array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
        '''
        edge_index_table = {}
        edge_index = 0
        for ri, col_indices in enumerate(self._row_list):
            for ci in col_indices:
                edge_index_table[ri, ci] = edge_index
                edge_index += 1

        edge_num = self.count_nonzero()
        cmatrix = Spmat([edge_num, edge_num])

        ri = 0
        for rj, col_indices in enumerate(self._row_list):
            for ci in col_indices:
                for cj in col_indices:
                    if ci != cj:
                        ei = edge_index_table[rj, cj]
                        cmatrix.entry(ri, ei)
                ri += 1

        return cmatrix

    def to_variable_node_matrix(self):
        r'''Returns a variable node matrix of the self parity check matrix

        see SumProductAlgorithm class

        Example:
            >>> import numpy as np
            >>> pcm = np.array([[1, 0, 1, 1, 1, 0, 0],
                                [1, 1, 0, 1, 0, 1, 0],
                                [0, 1, 1, 1, 0, 0, 1]])
            >>> spmat = make_spmat_from_ndarray(pcm)
            >>> spmat.to_variable_node_matrix().to_ndarray()
            array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
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
        '''
        edge_index_table = {}
        edge_index = 0
        for ri, col_indices in enumerate(self._row_list):
            for ci in col_indices:
                edge_index_table[ri, ci] = edge_index
                edge_index += 1

        edge_num = self.count_nonzero()
        vmatrix = Spmat([edge_num, edge_num])

        ri = 0
        for rj, col_indices in enumerate(self._row_list):
            for ci in col_indices:
                for rk in self._col_list[ci]:
                    if rj != rk:
                        ei = edge_index_table[rk, ci]
                        vmatrix.entry(ri, ei)
                ri += 1

        return vmatrix


def _read_spmat(in_):
    # line 1 : shape
    buf = in_.readline()
    shape = [int(s) for s in buf.split()[::-1]]
    matrix = Spmat(shape)

    # line 2--4 : unused
    for _ in range(3):
        in_.readline()

    # line 5-- : column indices
    for ri in range(shape[0]):
        buf = in_.readline()
        col_indices = [int(s) - 1  # 1-indexed -> 0-indexed
                       for s in buf.split()]
        for ci in col_indices:
            matrix.entry(ri, ci)

    return matrix


def read_spmat(in_):
    r'''read spmat file

        Args:
            in_ (string or TextIOWrapper): filename or file stream

        Returns:
            matrix (Spmat): the matrix defined by the given spmat file

        Example:
            >>> filename = 'hoge.spmat'
            >>>
            >>> # 1
            >>> hoge = read_spmat(filename)
            >>>
            >>> # 2
            >>> with open(filename, 'r') as f:
            >>>     hoge = read_spmat(f)
    '''
    if isinstance(in_, str):
        with open(in_, 'r') as f:
            return _read_spmat(f)
    else:
        return _read_spmat(in_)

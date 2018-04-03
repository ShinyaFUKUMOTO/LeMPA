import torch
import torch.nn as nn
from torch.autograd import Variable
import spmat

dtype = torch.FloatTensor


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class Code:
    '''This is a class to handle a error-correcting code defined by a given
    parity check matrix for message passing algorithms.

    Attributes:
        parity_check_matrix (Variable): the parity check matrix
        generator_matrix (Variable, optional): the generator matrix
        scatter_matrix (Variable): a matrix for scattering given
            log-likelifood ratios with edges of tanner graph
        gather_matrix (Variable): a matrix for calculating output from
            messages of check nodes
        check_node_matrix (Variable): a matrix for calculate message of check
            node
        variable_node_matrix (Variable): a matrix for calculating message of
            variable nodes

        code_length (int) -- the code length
        rate (double) -- the code rate
    '''

    def __init__(self, codedir=None, with_genmat=True):
        self._with_genmat = with_genmat

        if codedir is not None:
            self.load(codedir)

    def load(self, codedir, pcm_name='parity_check_matrix.spmat'):
        r'''load matrices from the code directory

        Args:
            codedir (string): directory name
            pcm_name (string): filename of spmat file

        Note:
            The code directry must contain spmat file which define a parity
            check matrix.
        '''
        codedir_ = codedir if codedir[-1] == '/' else codedir + '/'
        pcm_spmat = spmat.read_spmat(codedir_ + 'parity_check_matrix.spmat')
        pcm_ndarray = pcm_spmat.to_ndarray()

        factory = {
            'parity_check_matrix':
                lambda: pcm_ndarray,
            'generator_matrix':
                lambda: spmat.to_generator_matrix(pcm_ndarray),
            'scatter_matrix':
                lambda: pcm_spmat.to_scatter_matrix().to_ndarray(),
            'gather_matrix':
                lambda: pcm_spmat.to_gather_matrix().to_ndarray(),
            'variable_node_matrix':
                lambda: pcm_spmat.to_variable_node_matrix().to_ndarray(),
            'check_node_matrix':
                lambda: pcm_spmat.to_check_node_matrix().to_ndarray(),
        }

        def to_path(name): return codedir_ + name + '.dat'
        for name, maker in factory.items():
            if not self._with_genmat and name == 'generator_matrix':
                continue

            try:
                self.__dict__[name] = torch.load(to_path(name))
            except FileNotFoundError:
                self.__dict__[name] = Variable(torch.Tensor(maker()))

    def save(self, codedir):
        '''save a parity check matrix and related matrices

        Args:
            codedir (string) --
                a string containing a directry name for the matrices
        '''

        def path(name): return codedir + '/' + name + '.dat'
        torch.save(self.parity_check_matrix, path('parity_check_matrix'))
        if self._with_genmat and self.generator_matrix is not None:
            torch.save(self.generator_matrix, path('generator_matrix'))
        torch.save(self.gather_matrix, path('gather_matrix'))
        torch.save(self.scatter_matrix, path('scatter_matrix'))
        torch.save(self.check_node_matrix, path('check_node_matrix'))
        torch.save(self.variable_node_matrix, path('variable_node_matrix'))

    def encode(self, message):
        if isinstance(message, Variable):
            g = self.generator_matrix
        elif isinstance(message, torch.Tensor):
            g = self.generator_matrix.data
        else:
            error_message = r'''
                code.encode received an invalid argument -- got({}),
                but expected: torch.Tensor or torch.autograd.Variable
            '''.format(type(message))
            raise TypeError(error_message)

        return torch.mm(message, g).fmod(2)

    def parity_check(self, word):
        if isinstance(word, Variable):
            h = self.parity_check_matrix
        elif isinstance(word, torch.Tensor):
            h = self.parity_check_matrix.data
        else:
            error_message = r'''
                code.parity_check received an invalid argument -- got({}),
                but expected: torch.Tensor or torch.autograd.Variable
            '''.format(type(word))
            raise TypeError(error_message)

        eps = 1e-10
        syndrome = torch.mm(h, word.transpose(0, 1)).fmod(2)
        return (syndrome < eps).all()

    def cuda(self):
        for name, var in self.__dict__.items():
            if isinstance(var, Variable):
                self.__dict__[name] = var.cuda()

    @property
    def code_length(self):
        '''return the code length'''
        return self.parity_check_matrix.shape[1]

    @property
    def rate(self):
        '''return the code rate if the given parity check matrix is fulllank'''
        return self.dim / float(self.code_length)

    @property
    def dim(self):
        '''return the dimension of the code if the given parity check matrix is
        fulllank
        '''
        shape = self.parity_check_matrix.shape
        return shape[1] - shape[0]


class SumProductAlgorithm:
    r'''This is a class for (log-domain) sum-product algorithm

    This class has four methods for sum-product algorithm.
    1. scatter:
        Scatter given LLRs with edges of tanner graph.
        Use to add LLRs to messages from nodes of tanner graph.

    2. check_node_process:
        Calculate messages from check nodes

    3. variable_node_process:
        Calculate messages from variable nodes

    4. gather:
        For each variable nodes, calculate sum of messages from connected
        check nodes. Use to calculate the (tentative) output of the
        algorithm for each round.

    Edges of tanner graph and messages passed through the edges are indexed
    by position of corresponding non-zero element in a parity check matrix
    H as follows:
        H = [[1, 1, 0, 0],        [[0, 1, -, -],
             [0, 1, 1, 0],    ->   [-, 2, 3, -],
             [0, 0, 1, 1]]         [-, -, 4, 5]].
    '''

    def __init__(self, code):
        self._code = code
        num_of_nonzero = code.parity_check_matrix.nonzero().shape[0]
        self._num_of_messages = num_of_nonzero

    @property
    def num_of_messages(self):
        r'''Returns the number of messages, i.e. the number of edges in the
        tanner graph'''

        return self._num_of_messages

    def gather(self, message, gather_matrix=None):
        r'''For each variable nodes, calculate sum of messages from connected
        check nodes.

        Use to calculate the (tentative) output of the algorithm for each
        round.
        '''

        r'''Scatter given LLRs with edges of tanner graph.
            Use to add LLRs to messages from nodes of tanner graph.
        '''

        if gather_matrix is None:
            gather_matrix = self._code.gather_matrix

        return torch.mm(message, gather_matrix)

    def scatter(self, message, scatter_matrix=None):
        r'''Scatter given LLRs with edges of tanner graph.
            Use to add LLRs to messages from nodes of tanner graph.
        '''

        if scatter_matrix is None:
            scatter_matrix = self._code.scatter_matrix

        return torch.mm(message, scatter_matrix)

    def check_node_process(self, message, check_node_matrix=None):
        r'''Calculate messages from check nodes'''

        if check_node_matrix is None:
            check_node_matrix = self._code.check_node_matrix

        unsigned_message = torch.abs(torch.tanh(0.5 * message))
        unsigned_message = torch.exp(torch.mm(torch.log(unsigned_message),
                                              check_node_matrix))
        unsigned_message = 0.99 * unsigned_message.clamp(min=-1.0, max=1.0)
        unsigned_message = 2 * atanh(unsigned_message)

        is_negative = (-torch.sign(message + 1e-10) + 1) / 2
        num_of_negative_values = torch.mm(is_negative,
                                          check_node_matrix)
        sign = -2 * num_of_negative_values.fmod(2) + 1

        return sign * unsigned_message

    def variable_node_process(self, message, variable_node_matrix=None):
        r'''Calculate messages from variable nodes'''
        if variable_node_matrix is None:
            variable_node_matrix = self._code.variable_node_matrix

        return torch.mm(message, variable_node_matrix)


class MessageNormalizer(nn.Module):

    def __init__(self, in_features, init_mean=1.0, init_stddev=0.01):
        super(MessageNormalizer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features))
        self.init_mean = init_mean
        self.init_stddev = init_stddev
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(mean=self.init_mean, std=self.init_stddev)

    def forward(self, message):
        return self.weight * message

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + 'out_features=' + str(self.out_features) + ')'


class NeuralSumProductModel(nn.Module):

    def __init__(self, code, num_of_iteration,
                 llr_normalization=False,
                 variable_node_normalization=True,
                 check_node_normalization=True):
        super(NeuralSumProductModel, self).__init__()
        self._spa = SumProductAlgorithm(code)
        self._code = code
        self.num_of_iteration = num_of_iteration

        code_length = self._code.code_length
        num_of_messages = self._spa.num_of_messages

        self.llr_normalization = llr_normalization
        if llr_normalization:
            self.llr_normalizers = torch.nn.ModuleList(
                [MessageNormalizer(num_of_messages)
                    for _ in range(num_of_iteration)])

            self.llr_normalizers_for_output = torch.nn.ModuleList(
                [MessageNormalizer(code_length)
                    for _ in range(num_of_iteration)])

        self.variable_node_normalization = variable_node_normalization
        if variable_node_normalization:
            self.vnode_normalizers = torch.nn.ModuleList(
                [MessageNormalizer(num_of_messages)
                 for _ in range(num_of_iteration)])

        self.check_node_normalization = check_node_normalization
        if check_node_normalization:
            self.cnode_normalizers = torch.nn.ModuleList(
                [MessageNormalizer(num_of_messages)
                 for _ in range(num_of_iteration)])

    def forward(self, llr):
        spa = self._spa
        llr_list = [spa.scatter(llr)]
        extrinsic_value = Variable(torch.zeros(llr_list[-1].size()))
        output = []
        for i in range(self.num_of_iteration):
            if self.llr_normalization:
                llr_list += [self.llr_normalizers[i](llr_list[0])]
                llr_for_output = self.llr_normalizers_for_output[i](llr)
            else:
                llr_list += [llr_list[0]]
                llr_for_output = llr

            # variable node process
            a_priori_value = spa.variable_node_process(extrinsic_value)
            if self.variable_node_normalization:
                a_priori_value = self.vnode_normalizers[i](
                    a_priori_value)

            # check node process
            extrinsic_value = spa.check_node_process(
                a_priori_value + llr_list[-1])
            if self.check_node_normalization:
                extrinsic_value = self.cnode_normalizers[i](
                    extrinsic_value)

            # Temporary Decision
            temporary_output = spa.gather(extrinsic_value) + llr_for_output
            output.append(temporary_output)

        return output

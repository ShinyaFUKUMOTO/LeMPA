r'''Implement traditional sum-product decoder and evaluate performance '''

import sum_product as sp
import numpy as np
import torch
from torch.autograd import Variable
import time

cuda = False
seed = 1

# model
codedir = 'data/PEGREG504x1008'
decoding_iteration = 20  # number of iteration for decoding
all_zero_codeword = False

# SNR range for performance evaluation
snr_from = 1
snr_to = 5
snr_step = 0.5


def isnan(x):
    return x != x


def dtype(tensor):
    return tensor.cuda() if cuda else tensor


class NeuralSumProductModel(torch.nn.Module):

    def __init__(self, code, num_of_iteration):
        super(NeuralSumProductModel, self).__init__()
        self._spa = sp.SumProductAlgorithm(code)
        self._code = code
        self.num_of_iteration = num_of_iteration

    def forward(self, llr):
        spa = self._spa
        scattered_llr = spa.scatter(llr)

        extrinsic_value = Variable(dtype(torch.zeros(scattered_llr.size())))
        output = []
        for i in range(self.num_of_iteration):
            # variable node process
            a_priori_value = spa.variable_node_process(extrinsic_value)

            # check node process
            extrinsic_value = spa.check_node_process(
                a_priori_value + scattered_llr)

            # Temporary Decision
            temporary_output = spa.gather(extrinsic_value) + llr
            output.append(temporary_output)

        return output


def bin_to_bip(x):
    return -2 * x + 1


def bip_to_bin(x):
    return -0.5 * (x - 1)


def snr_to_var(snr, rate):
    return 0.5 * (1.0 / pow(10.0, snr / 10.0)) / rate


def calc_llr(x, var):
    return 2 * x / var


def eval_accuracy(code, model):
    decoder = model
    minibatch_size = 100
    code_length = code.code_length
    infoword_length = code.dim
    code_rate = code.rate
    mean = torch.zeros(minibatch_size, code_length)

    print('snr ser bler serr symbols blerr blocks rtime ptime')
    for snr in np.arange(snr_from, snr_to, snr_step):
        variance = snr_to_var(snr, code_rate)
        stddev = variance**0.5

        block_num = 0
        symbol_num = 0
        block_error_num = 0
        symbol_error_num = 0

        start_rtime = time.time()
        start_ptime = time.process_time()

        while True:
            if all_zero_codeword:
                codeword = dtype(torch.zeros([minibatch_size, code_length]))
            else:
                message = dtype(
                    torch.Tensor(
                        minibatch_size,
                        infoword_length).random_(0, 2))
                codeword = code.encode(message)

            transmitted_signal = bin_to_bip(codeword)
            channel_noise = dtype(torch.normal(mean, stddev))
            received_signal = transmitted_signal + channel_noise
            llr = calc_llr(received_signal, variance)

            soft_output = decoder(Variable(llr))[-1].data
            estimated_word = bip_to_bin(torch.sign(soft_output))

            error = torch.sum(torch.abs(codeword - estimated_word) > 0.5,
                              dim=1)
            block_error_num += torch.sum(error > 0)
            symbol_error_num += torch.sum(error)

            block_num += minibatch_size
            symbol_num += minibatch_size * code_length

            if symbol_error_num > 5000:
                break

        elapsed_rtime = time.time() - start_rtime
        elapsed_ptime = time.process_time() - start_ptime

        symbol_error_rate = symbol_error_num / float(symbol_num)
        block_error_rate = block_error_num / float(block_num)

        s = '{snr} {ser:.2e} {bler:.2e} {serr} {symbols} {blerr} {blocks}'\
            '{rtime:9.2e} {ptime:9.2e}'
        s = s.format(
            snr=snr,
            ser=symbol_error_rate, bler=block_error_rate,
            serr=symbol_error_num, symbols=symbol_num,
            blerr=block_error_num, blocks=block_num,
            rtime=elapsed_rtime, ptime=elapsed_ptime
        )
        print(s)


def main(cuda_=True):
    global cuda
    cuda = cuda_

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    print('Prepare matrices')
    use_genmat = not all_zero_codeword
    code = sp.Code(codedir, use_genmat)
    code.save(codedir)
    if cuda:
        code.cuda()

    print('  codelength = {}'.format(code.code_length))
    print('  rate = {}'.format(code.rate))

    model = NeuralSumProductModel(code, decoding_iteration)
    if cuda:
        model.cuda()

    print('eval decoding performance')
    eval_accuracy(code, model)


if __name__ == '__main__':
    main(False)

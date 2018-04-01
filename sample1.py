r'''a sample of training neural sum-product decoder

This script train a neural sum-product decoder for a randomly constructed
(3, 6)-LDPC code with length 1008 and rate R = 0.5.
The decoder has trainable normalization factors for each message in sum-product
decoding. Too small initial values are choosed to show the effectiveness of
training.

decoding performance before training:
    snr ser bler serr symbols blerr blocks rtime ptime
    1.0 1.14e-01 1.00e+00 11536 100800 100 100 5.39e-02  5.34e-02
    1.5 9.67e-02 1.00e+00 9747 100800 100 100 5.34e-02  5.34e-02
    2.0 8.13e-02 1.00e+00 8198 100800 100 100 5.00e-02  4.97e-02
    2.5 6.58e-02 1.00e+00 6628 100800 100 100 4.93e-02  4.40e-02
    3.0 5.02e-02 1.00e+00 5064 100800 100 100 4.95e-02  4.96e-02
    3.5 3.82e-02 1.00e+00 3851 100800 100 100 4.95e-02  4.97e-02

decoding performance after training at snr 3.0 with 1000 batchs (size 120):
    snr ser bler serr symbols blerr blocks rtime ptime
    1.0 8.44e-02 1.00e+00 8508 100800 100 100 3.57e-02  3.56e-02
    1.5 5.55e-02 1.00e+00 5592 100800 100 100 3.72e-02  3.72e-02
    2.0 2.48e-02 9.70e-01 2499 100800 97 100 3.81e-02  3.83e-02
    2.5 6.89e-03 7.45e-01 1389 201600 149 200 7.16e-02  7.18e-02
    3.0 1.14e-03 2.86e-01 1038 907200 257 900 3.22e-01  3.23e-01
    3.5 7.58e-05 3.18e-02 1001 13204800 416 13100 4.76e+00  4.76e+00

training log:
    step loss
    100 0.1147
    200 0.08904
    300 0.048918
    400 0.013416
    500 0.0074216
    600 0.0059614
    700 0.0064156
    800 0.0039823
    900 0.0047424
    1000 0.0033085
'''

import sum_product as sp
import numpy as np
import torch
from torch.autograd import Variable
import time

cuda = False
seed = 1

# model
codedir = 'data/PEGREG504x1008'
decoding_iteration = 5  # number of iteration for decoding
all_zero_codeword = True
init_mean = 0.3  # mean of initial values

# channel state for training
tsnr = 3

# SNR range for performance evaluation
snr_from = 1
snr_to = 4
snr_step = 0.5

# training parameters
batch_size = 120
batch_num = 100

log_interval = 10


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

        num_of_messages = self._spa.num_of_messages

        self.vnode_normalizers = torch.nn.ModuleList(
            [sp.MessageNormalizer(num_of_messages, init_mean=init_mean)
             for _ in range(num_of_iteration)])

        self.cnode_normalizers = torch.nn.ModuleList(
            [sp.MessageNormalizer(num_of_messages, init_mean=init_mean)
             for _ in range(num_of_iteration)])

    def forward(self, llr):
        spa = self._spa
        scattered_llr = spa.scatter(llr)

        extrinsic_value = Variable(dtype(torch.zeros(scattered_llr.size())))
        output = []
        for i in range(self.num_of_iteration):
            # variable node process
            a_priori_value = spa.variable_node_process(extrinsic_value)
            a_priori_value = self.vnode_normalizers[i](a_priori_value)

            # check node process
            extrinsic_value = spa.check_node_process(
                a_priori_value + scattered_llr)
            extrinsic_value = self.cnode_normalizers[i](extrinsic_value)

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


def train(code, model):
    code_length = code.code_length
    code_rate = code.rate
    infoword_length = code.dim

    mean = torch.zeros(batch_size, code_length)
    variance = snr_to_var(tsnr, code_rate)
    stddev = variance**0.5

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    start_rtime = time.time()
    start_ptime = time.process_time()

    for step in range(1, batch_num + 1):

        optimizer.zero_grad()

        if all_zero_codeword:
            codeword = torch.zeros([batch_size, code_length])
        else:
            infoword = torch.Tensor(batch_size, infoword_length).random_(0, 2)
            infoword = dtype(infoword)
            codeword = code.encode(infoword)
        codeword = Variable(dtype(codeword))

        transmitted_signal = bin_to_bip(codeword)
        channel_noise = Variable(dtype(torch.normal(mean, stddev)))
        received_signal = transmitted_signal + channel_noise
        llr = 2 * received_signal / variance

        output = model(llr)[-1]
        if isnan(output).any():
            continue

        loss = criterion(-output, codeword)
        loss.backward()

        grads = torch.stack([param.grad for param in model.parameters()])
        if isnan(grads).any():
            continue

        optimizer.step()

        if step % log_interval == 0:
            print('step loss')
            print('{step} {loss:.5}'.format(step=step, loss=loss.data[0]))

            print('weights')
            weights = [param.data for param in model.parameters()]
            print(torch.stack(weights))

    elapsed_rtime = time.time() - start_rtime
    elapsed_ptime = time.process_time() - start_ptime
    print('real time = {}[sec]'.format(elapsed_rtime))
    print('process time = {}[sec]'.format(elapsed_ptime))


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

            if symbol_error_num > 1000:
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

    print('eval accuracy')
    eval_accuracy(code, model)

    print('train')
    train(code, model)

    print('eval accuracy')
    eval_accuracy(code, model)


if __name__ == '__main__':
    main(False)

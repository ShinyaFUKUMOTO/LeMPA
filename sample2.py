import argparse
import sum_product as sp
import numpy as np
import time
import torch
from torch.autograd import Variable

description = 'Sample of training and evaluating a Neural sum-product decoder.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--codedir', type=str, default=None, required=True,
                    help='name of data directry')
parser.add_argument('--decoding-iteration', type=int, default='5', metavar='I',
                    help='number of decoding iterations')

tsnr_args = parser.add_argument_group('training SNR',
                                      'Channel SNR for training')
tsnr_args.add_argument('--tsnr-from', type=float, default=2.0,
                       help='start of training SNR interval')
tsnr_args.add_argument('--tsnr-to', type=float, default=6.0,
                       help='end of training SNR interval')
tsnr_args.add_argument('--tsnr-step', type=float, default=1.0,
                       help='spacing between training SNRs')

snr_args = parser.add_argument_group('SNR',
                                     'Channel SNR for evaluating performance')
snr_args.add_argument('--snr-from', type=float, default=2.0,
                      help='start of evaluating SNR interval')
snr_args.add_argument('--snr-to', type=float, default=6.0,
                      help='end of evaluating SNR interval')
snr_args.add_argument('--snr-step', type=float, default=1.0,
                      help='spacing between evaluating SNRs')

parser.add_argument('--sample-num', type=int, default=20, metavar='n',
                    help='number of samples for each tsnr (default: 20)')
parser.add_argument('--batch-num', type=int, default=1000, metavar='n',
                    help='number of batchs for training (default: 1000)')
parser.add_argument('--eval-batch-size', type=int, default=100, metavar='n',
                    help='input batch size for evaluating performance '
                         '(default: 100)')
parser.add_argument('--all-zero-codeword', action='store_true', default=False,
                    help='all-zero codeword assumption in training and '
                         'evaluating')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

is_use_zero_codeword = True


def isnan(x):
    return x != x


def dtype(tensor):
    return tensor.cuda() if args.cuda else tensor


class NeuralSumProductModel(torch.nn.Module):

    def __init__(self, code, num_of_iteration,
                 llr_normalization=False,
                 variable_node_normalization=True,
                 check_node_normalization=True):
        super(NeuralSumProductModel, self).__init__()
        self._spa = sp.SumProductAlgorithm(code)
        self._code = code
        self.num_of_iteration = num_of_iteration

        code_length = self._code.code_length
        num_of_messages = self._spa.num_of_messages

        self.llr_normalization = llr_normalization
        if llr_normalization:
            self.llr_normalizers = torch.nn.ModuleList(
                [sp.MessageNormalizer(num_of_messages)
                    for _ in range(num_of_iteration)])

            self.llr_normalizers_for_output = torch.nn.ModuleList(
                [sp.MessageNormalizer(code_length)
                    for _ in range(num_of_iteration)])

        self.variable_node_normalization = variable_node_normalization
        if variable_node_normalization:
            self.vnode_normalizers = torch.nn.ModuleList(
                [sp.MessageNormalizer(num_of_messages)
                 for _ in range(num_of_iteration)])

        self.check_node_normalization = check_node_normalization
        if check_node_normalization:
            self.cnode_normalizers = torch.nn.ModuleList(
                [sp.MessageNormalizer(num_of_messages)
                 for _ in range(num_of_iteration)])

    def forward(self, llr):
        spa = self._spa
        llr_list = [spa.scatter(llr)]
        extrinsic_value = Variable(dtype(torch.zeros(llr_list[-1].size())))
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


def bin_to_bip(x):
    return -2 * x + 1


def bip_to_bin(x):
    return -0.5 * (x - 1)


def snr_to_var(snr, rate):
    return 0.5 * (1.0 / pow(10.0, snr / 10.0)) / rate


def calc_llr(x, var):
    return 2 * x / var


def train(code, model, dumping_step=100):
    code_length = code.code_length
    code_rate = code.rate
    infoword_length = code.dim

    tsnrs = np.arange(args.tsnr_from, args.tsnr_to, args.tsnr_step)
    batch_size = args.sample_num * len(tsnrs)
    mean = Variable(dtype(torch.zeros(batch_size, code_length)))

    variance = []
    for tsnr in tsnrs:
        var = snr_to_var(tsnr, code_rate)
        for _ in range(args.sample_num):
            variance.append([var] * code_length)
    variance = Variable(dtype(torch.Tensor(variance)))
    stddev = torch.sqrt(variance)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for step in range(args.batch_num):

        optimizer.zero_grad()

        if args.all_zero_codeword:
            codeword = torch.zeros([batch_size, code_length])
        else:
            infoword = torch.Tensor(batch_size, infoword_length).random_(0, 2)
            infoword = dtype(infoword)
            codeword = code.encode(infoword)
        codeword = Variable(dtype(codeword))

        transmitted_signal = bin_to_bip(codeword)
        channel_noise = dtype(torch.normal(mean, stddev))
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

        if step % dumping_step == dumping_step - 1:
            print('{step} loss:{loss}'.format(
                step=step + 1, loss=loss.data[0]))

            pm = []
            for param in model.parameters():
                pm.append(param.data)
            # print(torch.stack(pm))


def eval_accuracy(code, model):
    decoder = model
    minibatch_size = args.eval_batch_size
    code_length = code.code_length
    infoword_length = code.dim
    code_rate = code.rate
    mean = torch.zeros(minibatch_size, code_length)

    print('snr ser bler serr symbols blerr blocks rtime ptime')
    for snr in np.arange(args.snr_from, args.snr_to, args.snr_step):
        variance = snr_to_var(snr, code_rate)
        stddev = variance**0.5

        block_num = 0
        symbol_num = 0
        block_error_num = 0
        symbol_error_num = 0

        start_rtime = time.time()
        start_ptime = time.process_time()

        while True:
            if args.all_zero_codeword:
                codeword = dtype(torch.zeros([minibatch_size, code_length]))
            else:
                message = dtype(
                    torch.Tensor(
                        minibatch_size,
                        infoword_length).random_(0, 2))
                codeword = code.encode(message)

            transmitted_signal = bin_to_bip(codeword)
            channel_noise = torch.normal(mean, stddev)
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

        s = '{snr} {ser:.2e} {bler:.2e} {serr} {symbols} {blerr} {blocks}' \
            '{rtime:9.2e} {ptime:9.2e}'
        s = s.format(
            snr=snr,
            ser=symbol_error_rate, bler=block_error_rate,
            serr=symbol_error_num, symbols=symbol_num,
            blerr=block_error_num, blocks=block_num,
            rtime=elapsed_rtime, ptime=elapsed_ptime
        )
        print(s)


def main():
    use_genmat = not args.all_zero_codeword
    code = sp.Code(codedir=args.codedir, with_genmat=use_genmat)
    if args.cuda:
        code.cuda()

    model = sp.NeuralSumProductModel(code, args.decoding_iteration,
                                     variable_node_normalization=False,
                                     check_node_normalization=True)
    if args.cuda:
        model.cuda()
    train(code, model)
    eval_accuracy(code, model)


if __name__ == '__main__':
    main()

import time
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5


def train(input_tensor,
          target_tensor,
          encoder,
          decoder,
          encoder_optim,
          decoder_optim,
          criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    decoder_hidden = decoder.initHidden()
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    input_length = input_tensor.size(0)
    output_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length,
                                  encoder.hidden_size,
                                  device=device)

    loss = 0.0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_force = True if random.random(
    ) < teacher_forcing_ratio else False

    if use_teacher_force:
        for di in range(output_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(output_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / target_tensor.size(0)


def trainIters(encoder,
               decode,
               n_iters,
               print_every=1000,
               plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(decode.parameters(), lr=learning_rate)
    training_pairs = [
        tensorsFromPair(random.choice(pairs)) for i in range(n_iters)
    ]
    criterion = nn.NLLLoss()

    for iter in trange(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder, decode,
                     encoder_optim, decoder_optim, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"[log] {time.time() - start} loss: {print_loss_avg}")
            # print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #  iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


######################################################################
# Evaluation
######################################################################


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length,
                                      encoder.hidden_size,
                                      device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size,
                                   output_lang.n_words,
                                   dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

    evaluateRandomly(encoder1, attn_decoder1)

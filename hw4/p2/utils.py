import numpy as np
import Levenshtein
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummaryX import summary

from params import *


def sanity(loader, model):

    for data in loader:
        x, lx, y, ly = data
        break

    print('x:', x.shape)
    print('lx:', lx.shape)
    print('y:', y.shape)
    print('ly:', ly.shape)

    print(model)
    summary(model,
            x.to(next(model.parameters()).device), lx,
            y.to(next(model.parameters()).device))


def plot_attention(epoch, attention):

    plt.close()
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.savefig('attention_plots/attempt_'+ATTEMPT_ID+'_epoch_'+str(epoch)+'.png')


def indices_to_chars(indices):

    tokens = []

    for i in indices:

        if VOCAB[int(i)] == VOCAB[SOS_TOKEN]:
            continue
        if VOCAB[int(i)] == VOCAB[EOS_TOKEN]:
            break

        tokens.append(VOCAB[int(i)])

    return tokens


def calc_edit_distance(predictions, y, ly, print_example=False):

    dist = 0
    batch_size = predictions.shape[0]

    for batch_idx in range(batch_size):

        y_sliced = indices_to_chars(y[batch_idx, 0:ly[batch_idx]])
        pred_sliced = indices_to_chars(predictions[batch_idx])

        y_string = ''.join(y_sliced)
        pred_string = ''.join(pred_sliced)

        dist += Levenshtein.distance(pred_string, y_string)

    if print_example:
        print('\nground truth :', y_string)
        print('prediction   :', pred_string)

    dist /= batch_size

    return dist


def make_output(pred):

    batch_size = pred.shape[0]
    predicted_strings = []

    for batch_idx in range(batch_size):

        pred_sliced = indices_to_chars(pred[batch_idx])
        pred_string = ''.join(pred_sliced)

        predicted_strings.append(pred_string)

    return predicted_strings


def tf_scheduler(epoch, tf_rate):

    if epoch < 15:
        return tf_rate

    return max(0.2, tf_rate - 0.05)

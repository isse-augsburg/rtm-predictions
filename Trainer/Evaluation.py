import os
from functools import partial
from multiprocessing.pool import Pool

from PIL import Image
from pathlib import Path
import numpy as np



def pixel_wise_loss_multi_input_single_label(input, target):
    print('Loss')
    loss = 0
    for el in input:
        out = el - target
        # out = out * weights.expand_as(out)
        loss += out.sum(0)
    return loss


def plot_predictions_and_label(input, target, _str):
    if os.name == 'nt':
        debug_path = Path(r'X:\s\t\stiebesi\code\debug\overfit')
    else:
        debug_path = Path('/cfs/home/s/t/stiebesi/code/debug/overfit/')
    (debug_path / 'predict').mkdir(parents=True, exist_ok=True)

    x = input.reshape(input.shape[0], 155, 155)
    x = x * 255
    with Pool() as p:
        p.map(partial(save_img, debug_path / 'predict', _str, x), range(0, input.shape[0], 1))
    y = target.reshape(target.shape[0], 155, 155)
    y = y * 255
    im = Image.fromarray(np.asarray(y[0]))
    path = debug_path / 'label'
    path.mkdir(parents=True, exist_ok=True)
    file = f'{_str}.png'
    im.convert('RGB').save(path / file)
    im.close()


def save_img(path, _str, x, index):
    try:
        im = Image.fromarray(np.asarray(x[index]))
        file = f'{_str}_{index}.png'
        im.convert('RGB').save(path / file)
        im.close()
    except KeyError:
        print('ERROR: save_img')


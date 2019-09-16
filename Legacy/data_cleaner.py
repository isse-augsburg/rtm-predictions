import os
from pathlib import Path

import h5py
from PIL import Image

if os.name == 'nt':
    data_root = Path(r'Y:\data\RTM\Lautern\output\with_shapes')
    savepath = Path(r'C:\Users\stiebesi\code\saved_models')
else:
    data_root = Path('/cfs/share/data/RTM/Lautern/output/with_shapes')
    savepath = Path('/cfs/home/s/t/stiebesi/code/saved_models')

path = data_root / '2019-06-05_15-30-52_1050p'


def search_broken_pngs(path):
    all_pngs = path.glob('**/*.png')
    # with Pool() as p:
    #     p.map(open_img, all_pngs)
    for e in all_pngs:
        try:
            im = Image.open(e)
            im.close()
        except IOError:
            print('Broken Image', e)
            e.unlink()


def search_very_short_erfh5_states(path):
    all_erfh5 = path.glob('**/*.erfh5')
    for e in all_erfh5:
        f = h5py.File(e, 'r')
        len_states = len(list(f['post']['singlestate'].keys()))
        if len_states < 5:
            print('Short:', e)
        f.close()


# def open_img(e):
#     try:
#         im = Image.open(e)
#         im.close()
#     except IOError:
#         print('Broken Image', e)


if __name__ == "__main__":
    search_very_short_erfh5_states(path)
    # search_broken_pngs(path)

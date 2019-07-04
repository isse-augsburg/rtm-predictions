import os

import numpy as np
from pathlib import Path
import time
import zipfile
import shutil

def derive_k1k2_from_fvc(fvcs):
    points = np.array([(0, 1e-6), (0.4669, 1.62609e-10), (0.5318, 5.14386e-11), (0.5518, 4.28494e-11)])
    x = points[:, 0]
    y = points[:, 1]
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)


def fvc_to_k1(fvc):
    return 1.00000000e-06 * np.exp(-1.86478393e+01*fvc)


def rounded_random(value, minClip):
    return round(float(value)/ minClip) * minClip


def zip_folder(path, batch_num, delete_after=True):
    print(f'Zipping {path} ...')
    t0 = time.time()
    ps = [x for x in Path(path).glob('**/*') if x.is_file()]
    zip_file = zipfile.ZipFile(str(path) + f'_{batch_num}.zip', 'w')
    with zip_file:
        for file in ps:
            zip_file.write(str(file), compress_type=zipfile.ZIP_DEFLATED)
    print(f'Zipping took {(time.time() - t0)/60:.1f} minutes.')
    if delete_after:
        print(f'Deleting {path} ...')
        shutil.rmtree(path)
    return str(path) + f'_{batch_num}.zip'


def convert_win_to_unix_lineendings(filename):
    f_read = open(filename, "r")
    fileContents = f_read.read()
    f_read.close()
    f_unix_line_endings = open(filename, "w", newline="\n")
    f_unix_line_endings.write(fileContents)
    f_unix_line_endings.close()


def zip_file(delete_after=True, path=''):
    zip_filename = path + '.zip'
    zip_file = zipfile.ZipFile(zip_filename, 'w')
    zip_file.write(path, compress_type=zipfile.ZIP_DEFLATED)
    if delete_after:
        os.remove(path)
    return zip_filename

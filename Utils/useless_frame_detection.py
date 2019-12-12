import sys
from multiprocessing.pool import Pool
from pathlib import Path

import h5py
import numpy as np
import tqdm


def mark_useless_frames_in_file(file):
    try:
        result_file = h5py.File(str(file), 'r')
        meta_file = h5py.File(str(file).replace("RESULT.erfh5", "meta_data.hdf5"), "r+")
    except OSError as e:
        print(e)
        return
    ignore_list = []
    keys = list(result_file["/post/singlestate"].keys())

    for i, k in enumerate(keys):
        try:
            z = result_file[f"/post/singlestate/{k}/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res"][()].flatten()
        except KeyError:
            continue
        ones = np.ones_like(z)
        filling_perc = np.sum(z) / np.sum(ones)
        if filling_perc >= 1.0:
            ignore_list.append(int(str(k).replace("state", "0")))
        result_file.close()
    #print(ignore_list)
    try:
        useless_states = meta_file.require_group('useless_states')
        useless_states.create_dataset('singlestates', data=np.array(ignore_list))
        meta_file.close()
    except RuntimeError:
        print(f"Group or dataset could not be created with file: {str(file)}")


def mark_useless_frames(root_dir):
    files = list(root_dir.rglob("*RESULT.erfh5"))

    with Pool() as p:
        l = list(tqdm.tqdm(p.imap_unordered(mark_useless_frames_in_file,files), total=len(files)))


if __name__ == "__main__":
    # data_root / "2019-07-24_16-32-40_5000p",  # X             # X             # .2 - .8   # High
    # data_root / '2019-11-08_15-40-44_5000p'  # X             # X             # .3 - .5   # Low
    mark_useless_frames(Path(sys.argv[1]))


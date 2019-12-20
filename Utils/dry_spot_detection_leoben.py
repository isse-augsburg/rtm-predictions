import io
import socket
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from time import time

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from Utils.img_utils import scale_coords_leoben


def __analyze_image(img, perm_map=None):
    _, threshold = cv2.threshold(img, 70, 190, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_size = 3
    dryspots = np.zeros_like(img, dtype=np.float)
    spots = False
    probs = []
    for i, cnt in enumerate(contours):
        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
        size = cv2.contourArea(cnt)
        if size < min_size:
            continue
        # max contour
        if size > 273440:
            continue

        empty = np.zeros_like(img, dtype=np.float)
        cv2.fillPoly(empty, [np.squeeze(approx)], 255)
        perm_cut = np.where(empty == 255, perm_map.astype(np.float), 0)  # take values from perm map, where contour is
        del empty
        # filter values of the background assuming background is between 70 and 65
        perm_cut = np.where((perm_cut <= 70) & (perm_cut >= 65), 0, perm_cut)
        perm_cut = np.where((perm_cut == 0), 0, 255)  # focus on anything other than background
        avg_dryspot_prob = np.sum(perm_cut, dtype=np.float) / size  # normalize with size of contour area
        # print(avg_dryspot_prob, np.sum(perm_cut,dtype=np.float), size) # debug print statement
        del perm_cut
        probs.append(avg_dryspot_prob)
        if avg_dryspot_prob > 250:
            cv2.fillPoly(dryspots, [np.squeeze(approx)], 255)
            spots = True
    del contours, threshold
    del perm_map

    return spots, dryspots, probs


def dry_spot_analysis(file_path, output_dir_imgs, triang, Xi, Yi, xi, yi, change_meta_file=False,
                      save_flowfront_img=False, silent=False, detect_useless=False):
    try:
        f = h5py.File(file_path, "r")
    except OSError:
        print('ERROR: Could not open file(s)!', file_path)
        return

    t00 = time()
    output_dir_imgs.mkdir(exist_ok=True, parents=True)

    keys = list(f["/post/singlestate"].keys())

    # Fiber fraction map creation with tripcolor
    fvc = f["/post/constant/entityresults/SHELL/FIBER_FRACTION/ZONE1_set1/erfblock/res"][()].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tripcolor(triang, fvc, cmap="gray")
    ax.set(xlim=(0, 375), ylim=(0, 300))
    plt.axis("off")
    plt.tight_layout()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.clim(0, 1)
    del fvc

    perm_bytes = io.BytesIO()
    plt.savefig(perm_bytes, bbox_inches=extent)
    fig.clear()
    plt.cla()
    plt.close(fig)

    perm_bytes.seek(0)
    file_bytes = np.asarray(bytearray(perm_bytes.read()), dtype=np.uint8)
    perm_bytes.close()
    perm_map = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    del file_bytes
    del perm_bytes

    spot_list_s = []
    spot_list_e = []
    b_set = False
    min_count_of_consecutive_dryspots = 2
    consecutive_dryspots = 0
    max_prob_old = 0
    spot_t = 0
    deltas_prob = []
    ignore_list = []
    for i, k in enumerate(keys):
        try:
            z = f[f"/post/singlestate/{k}/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res"][()].flatten()
        except KeyError:
            continue
        ones = np.ones_like(z)
        filling_perc = np.sum(z) / np.sum(ones)
        if (filling_perc >= 1.0):
            ignore_list.append(int(str(k).replace("state", "0")))
        interpolator = tri.LinearTriInterpolator(triang, z)
        zi = interpolator(Xi, Yi)
        del interpolator

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.contourf(xi, yi, zi, levels=10, cmap="gray", extend="both")
        del zi
        ax2.set(xlim=(0, 375), ylim=(0, 300))
        plt.axis("off")
        plt.tight_layout()
        extent = ax2.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())

        if save_flowfront_img:
            plt.savefig(str(output_dir_imgs / f"{k}_ff.png"), bbox_inches=extent)
            plt.cla()
            fig2.clf()
            plt.close(fig2)
            plt.close()
            img = cv2.imread(str(output_dir_imgs / f"{k}_ff.png"), cv2.IMREAD_GRAYSCALE)
        else:
            bytes_tmp = io.BytesIO()
            plt.savefig(bytes_tmp, bbox_inches=extent)
            fig2.clear()
            plt.close(fig2)
            plt.close()
            bytes_tmp.seek(0)
            file_bytes2 = np.asarray(bytearray(bytes_tmp.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)
            del file_bytes2
            bytes_tmp.close()
            del bytes_tmp
        img = 255 - img

        spot_b, dryspot_img, probs = __analyze_image(img, perm_map)
        del img

        if len(probs) > 0:
            # Saving the course of the maximum of avg. probabilities of dry spot
            max_prob = max(probs)
            delta_prob = max_prob_old - max_prob
            if abs(delta_prob) > 20:
                deltas_prob.append((abs(delta_prob), i + 1, k))
            max_prob_old = max_prob
        if spot_b:
            # Skipping dry spots that last very short
            consecutive_dryspots += 1
            if consecutive_dryspots >= min_count_of_consecutive_dryspots:
                if i + 1 != spot_t + 1:
                    spot_list_s.append(i + 1)
                    b_set = True
                spot_t = i + 1

        elif b_set:
            b_set = False
            spot_list_e.append(i + 1)

        cv2.imwrite(str(output_dir_imgs / (f"{k}_dry.png")), dryspot_img)
        del dryspot_img

    del spot_b, probs
    del perm_map

    if len(spot_list_e) < len(spot_list_s):
        spot_list_e.append(len(keys))

    print(ignore_list)
    if change_meta_file:
        try:
            meta_file = h5py.File(str(file_path).replace("RESULT.erfh5", "meta_data.hdf5"), "r+")
        except OSError:
            print('ERROR: Could not open file(s)!', str(file_path).replace("RESULT.erfh5", "meta_data.hdf5"))
            return
        states = []
        for i, key in enumerate(keys, 1):
            for start, stop in zip(spot_list_s, spot_list_e):
                if int(start) <= i < int(stop):
                    states.append(int(key.replace("state", "0")))

        # -> Moved to useless_frame_detection.py
        if detect_useless:
            try:
                useless_states = meta_file.require_group('useless_states')
                useless_states.create_dataset('singlestates', data=np.array(ignore_list))
                dry_group = meta_file.require_group('dryspot_states')
                dry_group.create_dataset('singlestates', data=np.array(states))
                meta_file.close()
            except RuntimeError:
                pass

    f.close()
    del f
    if not silent:
        print(
            f"{output_dir_imgs} Overall time: {time() - t00}. Remember: arrays start at one. "
            f'Dryspots at: {[f"{one} - {two}" for (one, two) in zip(spot_list_s, spot_list_e)]}, {deltas_prob[2:]}, '
            f'num of states {len(keys)}'
        )
    del keys
    return spot_list_s, spot_list_e, deltas_prob


def multiprocess_wrapper(triang, Xi, Yi, xi, yi, curr_path, i):
    date, time, _ = curr_path.split('_')
    stamp = date + '_' + time
    if socket.gethostname() == "swtse130":
        source = Path(r"X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes")
        output = Path(r"Y:\cache\DrySpotDet2")
    else:
        source = Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes")
        output = Path("/cfs/share/cache/DrySpotDet2")

    a, b, c = dry_spot_analysis(source / curr_path / str(i) / f"{stamp}_{i}_RESULT.erfh5", output / curr_path / str(i),
                                triang, Xi, Yi, xi, yi, change_meta_file=True, save_flowfront_img=True, detect_useless=True)


def create_triangle_mesh(file_path):
    f = h5py.File(file_path, "r")
    coord_as_np_array = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"][()]
    _all_coords = coord_as_np_array[:, :-1]
    scaled_coords = scale_coords_leoben(_all_coords)
    x = scaled_coords[:, 0]
    y = scaled_coords[:, 1]
    triangles = f["/post/constant/connectivities/SHELL/erfblock/ic"][()]
    triangles = triangles - triangles.min()
    triangles = triangles[:, :-1]
    xi = np.linspace(0, 375, 376)
    yi = np.linspace(0, 300, 301)
    Xi, Yi = np.meshgrid(xi, yi)
    triang = tri.Triangulation(x, y, triangles=triangles)
    return Xi, Yi, triang, xi, yi


def main_for_end():
    file_path = Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p/0/"
                     "2019-07-23_15-38-08_0_RESULT.erfh5")
    Xi, Yi, triang, xi, yi = create_triangle_mesh(file_path)
    curr_path = '2019-08-24_11-51-48_5000p'
    date, time, _ = curr_path.split('_')
    stamp = date + '_' + time
    source = Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes")
    output = Path("/cfs/share/cache/DrySpotDet2")

    a, b, c = dry_spot_analysis(source / curr_path / str(0) / f"{stamp}_{0}_RESULT.erfh5", output / curr_path / str(0),
                                triang, Xi, Yi, xi, yi, change_meta_file=False, save_flowfront_img=True)


def main():
    if socket.gethostname() == "swtse130":
        file_path = Path(r"X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-09-06_17-03-51_10000p\0"
                         r"\2019-09-06_17-03-51_0_RESULT.erfh5")
    else:
        file_path = Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-11_15-14-48_100p/0"
                         "/2019-07-11_15-14-48_0_RESULT.erfh5")
    Xi, Yi, triang, xi, yi = create_triangle_mesh(file_path)

    curr_path = '2019-11-29_16-56-17_10000p'
    num_runs = int(curr_path.split('_')[-1][:-1])

    with Pool() as p:
        p.map(partial(multiprocess_wrapper, triang, Xi, Yi, xi, yi, curr_path), range(0, num_runs))


if __name__ == "__main__":
    main()

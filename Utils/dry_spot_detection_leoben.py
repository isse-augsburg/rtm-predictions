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

from Utils.data_utils import scale_coords_leoben


def __analyze_image(img: np.ndarray, perm_map: np.ndarray):
    """
       Args:
           img (np.ndarray): array that contains the current flow front
           perm_map ( np.ndarray): array that contains the permeability map

       Returns:
             spots (bool): true if the img contains dryspots
             dryspots (np.array): a array containg the dryspots
             probs (list): a list containing the probabilities of the found dryspots.

        Finds contours within a image and overlays the resulting image with a permeability map.
        if there is a overlap, there is probably a dryspot.

       """
    _, threshold = cv2.threshold(img, 70, 190, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_size = 3
    dryspots = np.zeros_like(img, dtype=np.float)
    spots = False
    probs = []
    for i, cnt in enumerate(contours):
        # create a polygon from a contour with tolerance
        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
        size = cv2.contourArea(cnt)
        # if the contour is to small, ignore it
        if size < min_size:
            continue
        # if the contour contains the whole image, it can be ignored as well
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


def dry_spot_analysis(file_path, triang: tri.Triangulation, Xi: np.ndarray, Yi: np.ndarray, xi: np.ndarray,
                      yi: np.ndarray, change_meta_file=False, save_flowfront_img=False, output_dir_imgs=None,
                      silent=False, detect_useless=False):
    """
           Args:
                save_flowfront_img: if true, saves all intermediate image representations to the output_dir_imgs
                silent (bool): mute debug output
                detect_useless (bool):  frames that are 100% filled are not usefull for training. This function can mark
                                        these frames as useless and add them to the metadata file
                change_meta_file (bool): if true, writes dryspots and useless frames into the meta file
                yi: see create_triangle_mesh
                xi: see create_triangle_mesh
                Yi: see create_triangle_mesh
                Xi: see create_triangle_mesh
                triang: see create_triangle_mesh
                output_dir_imgs: A output folder if the images should be saved
                file_path (Path): a erfh5 file which is checked for dryspots
           Returns:
                spot_list_s (list): the starting points of time windows with dryspots
                spot_list_e (list): endpoints of time windows wiht dryspots. spotlist_s[2] - spotlist_e[2] would be
                                    the third dryspot window
                deltas_prob (list): contains big jumps in probability of dryspot during a run
           """
    try:
        f = h5py.File(file_path, "r")
    except OSError:
        print('ERROR: Could not open file(s)!', file_path)
        return

    t00 = time()
    if save_flowfront_img:
        output_dir_imgs.mkdir(exist_ok=True, parents=True)

    keys = list(f["/post/singlestate"].keys())
    # Fiber fraction map creation with tripcolor
    _ = __create_permeability_map(f, triang, colored=True,
                                  path=str(output_dir_imgs / f"permeability_map.png"))
    perm_map = __create_permeability_map(f, triang)
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
        zi = __interpolate_flowfront(Xi, Yi, ignore_list, k, triang, z)
        img = __create_flowfront_img(k, output_dir_imgs, save_flowfront_img, xi, yi, zi)

        spot_b, dryspot_img, probs = __analyze_image(img, perm_map)
        if save_flowfront_img:
            cv2.imwrite(str(output_dir_imgs / (f"{k}_permeability_map.png")), perm_map)
            cv2.imwrite(str(output_dir_imgs / (f"{k}_dry.png")), dryspot_img)

        # check for large jumps in dryspot probability. This is used to determine whether a file should be blacklisted.
        if len(probs) > 0:
            # Saving the course of the maximum of avg. probabilities of dry spot
            max_prob = max(probs)
            delta_prob = max_prob_old - max_prob
            if abs(delta_prob) > 20:
                deltas_prob.append((abs(delta_prob), i + 1, k))
            max_prob_old = max_prob
        # if there is a dryspot spotted within one of the frames, mark the start
        if spot_b:
            # Skipping dry spots that last very short
            consecutive_dryspots += 1
            if consecutive_dryspots >= min_count_of_consecutive_dryspots:
                if i + 1 != spot_t + 1:
                    spot_list_s.append(i + 1)
                    b_set = True
                spot_t = i + 1
        # if there was no dryspot detected, check if a dryspot was detected earlier. if so, mark the end.
        elif b_set:
            b_set = False
            consecutive_dryspots = 0
            spot_list_e.append(i + 1)

    if len(spot_list_e) < len(spot_list_s):
        spot_list_e.append(len(keys))

    if not silent:
        print(ignore_list)
    if change_meta_file:
        try:
            meta_file = h5py.File(str(file_path).replace("RESULT.erfh5", "meta_data.hdf5"), "r+")
            __update_meta_data(meta_file, spot_list_e, spot_list_s, ignore_list, detect_useless, keys)
        except OSError:
            print('ERROR: Could not open file(s)!', str(file_path).replace("RESULT.erfh5", "meta_data.hdf5"))
            return

    f.close()
    if not silent:
        print(
            f"{output_dir_imgs} Overall time: {time() - t00}. Remember: arrays start at one. "
            f'Dryspots at: {[f"{one} - {two}" for (one, two) in zip(spot_list_s, spot_list_e)]}, {deltas_prob[2:]}, '
            f'num of states {len(keys)}'
        )
    return spot_list_s, spot_list_e, deltas_prob


def __interpolate_flowfront(Xi, Yi, ignore_list, current_index, triang, values):
    ones = np.ones_like(values)
    filling_perc = np.sum(values) / np.sum(ones)
    if filling_perc >= 1.0:
        ignore_list.append(int(str(current_index).replace("state", "0")))
    interpolator = tri.LinearTriInterpolator(triang, values)
    # the PAM-RTM uses a triangle grid for filling states. Interpolate values over triangle grid with matplotlib
    zi = interpolator(Xi, Yi)
    return zi


def __update_meta_data(meta_file, spot_list_e, spot_list_s, ignore_list, detect_useless, keys):
    states = []
    for i, key in enumerate(keys, 1):
        for start, stop in zip(spot_list_s, spot_list_e):
            if int(start) <= i < int(stop):
                states.append(int(key.replace("state", "0")))
    if detect_useless:
        try:
            useless_states = meta_file.require_group('useless_states')
            useless_states.create_dataset('singlestates', data=np.array(ignore_list))
            dry_group = meta_file.require_group('dryspot_states')
            dry_group.create_dataset('singlestates', data=np.array(states))
            meta_file.close()
        except RuntimeError:
            pass


def __create_flowfront_img(k, output_dir_imgs, save_flowfront_img, xi, yi, zi):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(xi, yi, zi, levels=10, extend="both")  # cmap="gray")
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
    return img


def __create_permeability_map(f, triang, colored=False, path=None):
    fvc = f["/post/constant/entityresults/SHELL/FIBER_FRACTION/ZONE1_set1/erfblock/res"][()].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if colored:
        plt.tripcolor(triang, fvc)
    else:
        plt.tripcolor(triang, fvc, cmap="gray")
    ax.set(xlim=(0, 375), ylim=(0, 300))
    plt.axis("off")
    plt.tight_layout()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.clim(0, 1)
    del fvc
    perm_bytes = io.BytesIO()
    if path is not None:
        plt.savefig(path, bbox_inches=extent)
    plt.savefig(perm_bytes, bbox_inches=extent)
    fig.clear()
    plt.cla()
    plt.close(fig)
    perm_bytes.seek(0)
    file_bytes = np.asarray(bytearray(perm_bytes.read()), dtype=np.uint8)
    perm_bytes.close()
    if colored:
        perm_map = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        perm_map = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return perm_map


def multiprocess_wrapper(triang, Xi, Yi, xi, yi, curr_path, i):
    date, time, _ = curr_path.split('_')
    stamp = date + '_' + time
    # if socket.gethostname() == "swtse130":
    #     source = Path(r"X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes")
    #     output = Path(r"Y:\cache\DrySpotDet2")
    # else:
    source = Path("/cfs/share/data/RTM/Leoben/sim_output/")
    output = Path("/cfs/share/cache/DrySpotDet2")

    a, b, c = dry_spot_analysis(source / curr_path / str(i) / f"{stamp}_{i}_RESULT.erfh5",
                                triang, Xi, Yi, xi, yi,
                                output_dir_imgs=output / curr_path / str(i),
                                change_meta_file=True,
                                save_flowfront_img=True,
                                detect_useless=True)


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


def main():
    if socket.gethostname() == "swtse130":
        file_path = Path(r"X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-09-06_17-03-51_10000p\0"
                         r"\2019-09-06_17-03-51_0_RESULT.erfh5")
    else:
        file_path = Path("/cfs/share/data/RTM/Leoben/sim_output/2019-07-23_15-38-08_5000p/0/"
                         "2019-07-23_15-38-08_0_RESULT.erfh5")
    Xi, Yi, triang, xi, yi = create_triangle_mesh(file_path)

    curr_path = "2019-09-06_17-03-51_10000p"
    num_runs = int(curr_path.split('_')[-1][:-1])

    with Pool() as p:
        p.map(partial(multiprocess_wrapper, triang, Xi, Yi, xi, yi, curr_path), range(0, num_runs))


if __name__ == "__main__":
    main()

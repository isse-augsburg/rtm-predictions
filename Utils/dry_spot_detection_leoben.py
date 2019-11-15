import io
import socket
from multiprocessing import Pool
from pathlib import Path
from time import time

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def scale_coords_leoben(input_coords):
    scaled_coords = input_coords * 10
    return scaled_coords


def __analyze_image(img, perm_map=None):
    _, threshold = cv2.threshold(img, 70, 190, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_size = 3
    dryspots = np.zeros_like(img, dtype=np.float)
    spots = False
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
        # (x, y, w, h) = cv2.boundingRect(approx)
        # cv2.rectangle(img, (x,y), (x+w,y+h), 100, 2)

        size = cv2.contourArea(cnt)
        if size < min_size:
            continue
        # max countour
        if size > 273440:
            continue

        empty = np.zeros_like(img, dtype=np.float)
        cv2.fillPoly(empty, [np.squeeze(approx)], 255)
        perm_cut = np.where(empty == 255, perm_map.astype(np.float), 0)  # take values from perm map, where contour is
        perm_cut = np.where(
            (perm_cut <= 70) & (perm_cut >= 65), 0, perm_cut
        )  # filter values of the background assuming background is between 70 and 65
        perm_cut = np.where((perm_cut == 0), 0, 255)  # focus on anything other than background
        avg_dryspot_prob = np.sum(perm_cut, dtype=np.float) / size  # normalize with size of contour area
        # print(avg_dryspot_prob, np.sum(perm_cut,dtype=np.float), size) # debug print statement
        print(avg_dryspot_prob)
        if avg_dryspot_prob > 200:
            cv2.fillPoly(dryspots, [np.squeeze(approx)], 255)
            spots = True

    return spots, dryspots


def dry_spot_analysis(file_path, output_dir_imgs):
    try:
        f = h5py.File(file_path, "r")
        meta_file = h5py.File(str(file_path).replace("RESULT.erfh5", "meta_data.hdf5"), "r+")
    except OSError:
        return

    t00 = time()
    output_dir_imgs.mkdir(exist_ok=True, parents=True)

    coord_as_np_array = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"][()]
    _all_coords = coord_as_np_array[:, :-1]
    scaled_coords = scale_coords_leoben(_all_coords)
    x = scaled_coords[:, 0]
    y = scaled_coords[:, 1]
    spot_t = 0

    triangles = f["/post/constant/connectivities/SHELL/erfblock/ic"][()]
    triangles = triangles - triangles.min()
    triangles = triangles[:, :-1]
    xi = np.linspace(0, 375, 376)
    yi = np.linspace(0, 300, 301)
    Xi, Yi = np.meshgrid(xi, yi)
    triang = tri.Triangulation(x, y, triangles=triangles)

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

    perm_bytes = io.BytesIO()
    plt.savefig(perm_bytes, bbox_inches=extent)
    plt.close()
    perm_bytes.seek(0)
    file_bytes = np.asarray(bytearray(perm_bytes.read()), dtype=np.uint8)
    perm_map = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    spot_list_s = []
    spot_list_e = []
    b_set = False
    for i, k in enumerate(keys):
        print(i)
        try:
            data = f[f"/post/singlestate/{k}/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res"][()]
        except KeyError:
            continue
        z = data.flatten()

        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        interpolator = tri.LinearTriInterpolator(triang, z)
        zi = interpolator(Xi, Yi)
        ax2.contourf(xi, yi, zi, levels=10, cmap="gray", extend="both")
        ax2.set(xlim=(0, 375), ylim=(0, 300))
        plt.axis("off")
        plt.tight_layout()
        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        bytes_tmp = io.BytesIO()
        plt.savefig(bytes_tmp, bbox_inches=extent)
        plt.close()
        bytes_tmp.seek(0)
        file_bytes = np.asarray(bytearray(bytes_tmp.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        img = 255 - img
        spot_b, dryspot_img = __analyze_image(img, perm_map)
        if spot_b:
            if i + 1 != spot_t + 1:
                spot_list_s.append(i + 1)
                b_set = True
            spot_t = i + 1

        elif b_set:
            b_set = False
            spot_list_e.append(i + 1)

        cv2.imwrite(str(output_dir_imgs / (f"{k}_dry.png")), dryspot_img)

    if len(spot_list_e) < len(spot_list_s):
        spot_list_e.append(len(keys))

    states = []
    for i, key in enumerate(keys, 1):
        for start, stop in zip(spot_list_s, spot_list_e):
            if int(start) <= i < int(stop):
                states.append(int(key.replace("state", "0")))

    try:
        dry_group = meta_file.require_group('dryspot_states')
        dry_group.create_dataset('singlestates', data=np.array(states))
        meta_file.close()
    except RuntimeError:
        pass

    print(
        f"{output_dir_imgs} Overall time: {time() - t00}. Remember: arrays start at one."
        f'Dryspots at: {[f"{one} - {two}" for (one,two) in zip(spot_list_s,spot_list_e)]}'
    )

    return spot_list_s, spot_list_e


def multiprocess_wrapper(i):
    dry_spot_analysis(
        source / str(i) / str("2019-07-29_10-45-18_%d_RESULT.erfh5" % i),
        Path("/cfs/share/cache/DrySpotDet/2019-07-29_10-45-18_5000p") / str(i),
    )


if __name__ == "__main__":
    if socket.gethostname() == "swtse130":
        source = Path(r"X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-07-23_15-38-08_5000p")
    else:
        source = Path(
            "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/"
            "Leoben/output/with_shapes/2019-07-23_15-38-08_5000p"
        )
        source = Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-29_10-45-18_5000p")

    with Pool() as p:
        p.map(multiprocess_wrapper, range(0, 5000))

import colorsys
import multiprocessing
import time
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image, ImageDraw


def with_matplotlib():
    t0 = time.time()
    f = h5py.File(r'2019-05-17_16-45-57_0_RESULT.erfh5', 'r')
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    _all_coords = coord_as_np_array[:, :-1]
    x = _all_coords[:, 0]
    y = _all_coords[:, 1]

    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1] - 1

    fvc = f['post/constant/entityresults/SHELL/FIBER_FRACTION/ZONE1_set1/erfblock/res'].value

    states = list(f['post']['singlestate'].keys())

    for state in states:
        fig = plt.figure(figsize=(10, 10))
        t1 = time.time()
        ax = plt.axes()
        # ax.set_aspect('equal')
        filling = \
            f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
        mean_of_neighbour_nodes = filling[triangle_coords].reshape(len(triangle_coords), 3).mean(axis=1)
        tpc = ax.tripcolor(x, y, triangle_coords, facecolors=mean_of_neighbour_nodes, edgecolors='none')
        # plt.show()
        ax.set_axis_off()
        # ax.set_frame_on(False)
        plt.savefig(r'img\%s.png' % state, dpi=100)
        plt.close()
        print(time.time() - t1)


    # plt.show()
    # triang = mtri.Triangulation(x, y, triangles=triangle_coords)

    # tpc = ax.tripcolor(x, y, triangle_coords, facecolors=fvc.flatten(), edgecolors='k')
    # ax.set_title('Fiber content')

    # ax.triplot(triang, 'k-')

    # plt.show()
    print(time.time()-t0)


def with_pil():
    f = h5py.File(r'2019-05-17_16-45-57_0_RESULT.erfh5', 'r')
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    _all_coords = coord_as_np_array[:, :-1]
    scaled_coords = (_all_coords + 23.25) * 10

    fvc = f['post/constant/entityresults/SHELL/FIBER_FRACTION/ZONE1_set1/erfblock/res'].value
    perm_k1 = f['post/constant/entityresults/SHELL/PERMEABILITY1/ZONE1_set1/erfblock/res'].value
    perm_k2 = f['post/constant/entityresults/SHELL/PERMEABILITY2/ZONE1_set1/erfblock/res'].value
    thickness = f['post/constant/entityresults/SHELL/THICKNESS/ZONE1_set1/erfblock/res'].value

    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1] - 1
    states = list(f['post']['singlestate'].keys())

    fillings = []
    for state in states:
        fillings.append(f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                'res'][()])
    f.close()

    # draw_polygon_map(9999, fvc, scaled_coords, triangle_coords, colored=False)
    # exit()
    with Pool() as p:
        p.map(partial(plot_wrapper, triangle_coords, scaled_coords, fillings), range(len(fillings)))


def plot_wrapper(triangle_coords, scaled_coords, fillings, index):
    filling = fillings[index]
    means_of_neighbour_nodes = filling[triangle_coords].reshape(len(triangle_coords), 3).mean(axis=1)
    draw_polygon_map(index, means_of_neighbour_nodes, scaled_coords, triangle_coords, colored=False)


def draw_polygon_map(index, values_for_triangles, scaled_coords, triangle_coords, colored=True):
    mode = 'RGB' if colored else 'L'
    im = Image.new(mode, (465, 465))
    draw = ImageDraw.Draw(im)
    for i in range(len(triangle_coords)):
        val = values_for_triangles[i]
        if not colored:
            draw.polygon(scaled_coords[triangle_coords[i]], fill=(int(val * 255)))
        else:
            if val == 0.0:
                draw.polygon(scaled_coords[triangle_coords[i]], fill=(255, 0, 0))
            elif val == 1.0:
                draw.polygon(scaled_coords[triangle_coords[i]], fill=(0, 102, 255))
            else:
                h = 3.6 * val
                col = tuple(int(round(i * 255)) for i in colorsys.hsv_to_rgb(h, 1, 1))
                draw.polygon(scaled_coords[triangle_coords[i]], fill=col)
    if index % 10 == 0: print(index)
    im.save(Path(r'img3/%d.png' % index))


if __name__== "__main__":
    t0 = time.time()
    with_pil()
    print('Overall:', time.time() - t0)
    # with_matplotlib()
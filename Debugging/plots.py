import colorsys
import time

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image, ImageDraw


def with_matplotlib():
    t0 = time.time()
    f = h5py.File(r'C:\Users\stiebesi\Desktop\2019-05-17_16-45-57_0_RESULT.erfh5', 'r')
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
    # Create empty black canvas
    im = Image.new('L', (465, 465))

    # Draw red and yellow triangles on it and save
    draw = ImageDraw.Draw(im)
    # draw.polygon([(20, 10), (200, 200), (100, 20)], fill=(255, 0, 0))
    # draw.polygon([(200, 10), (200, 200), (150, 50)], fill='yellow')

    f = h5py.File(r'C:\Users\stiebesi\Desktop\2019-05-17_16-45-57_0_RESULT.erfh5', 'r')
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    _all_coords = coord_as_np_array[:, :-1]
    x = _all_coords[:, 0]
    y = _all_coords[:, 1]

    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1] - 1
    triang = mtri.Triangulation(x, y, triangles=triangle_coords)
    xxx = (_all_coords + 23.25) * 10
    states = list(f['post']['singlestate'].keys())

    for state in states:
        t0 = time.time()
        filling = \
            f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                'res'][()]
        means_of_neighbour_nodes = filling[triangle_coords].reshape(len(triangle_coords), 3).mean(axis=1)
        for i in range(len(triangle_coords)):
            # draw.polygon(xxx[triangle_coords[i]], fill=means_of_neighbour_nodes[i])
            # print(means_of_neighbour_nodes[i])
            val = means_of_neighbour_nodes[i]
            # if val == 0.0:
            #     draw.polygon(xxx[triangle_coords[i]], fill=(255, 0, 0))
            # elif val == 1.0:
            #     draw.polygon(xxx[triangle_coords[i]], fill=(0, 102, 255))
            # else:
            #     h = 3.6 * val
            #     col = tuple(int(round(i * 255)) for i in colorsys.hsv_to_rgb(h, 1, 1))
            #     draw.polygon(xxx[triangle_coords[i]], fill=col)

            # grey
            draw.polygon(xxx[triangle_coords[i]], fill=(int(val * 255)))
            # if val == 0.0:
            #     draw.polygon(xxx[triangle_coords[i]], fill=(0))
            # elif val > 0.0:
            #     draw.polygon(xxx[triangle_coords[i]], fill=(1000))

            # draw.polygon(xxx[triangle_coords[i]], fill=(int(val * 1000)))
        # im.show()
        im.save(r'img3\%s.png' % state)
        print(time.time() - t0)


if __name__== "__main__":
    with_pil()
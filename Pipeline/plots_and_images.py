import colorsys

import numpy as np
from PIL import Image, ImageDraw


def scale_coords_lautern(input_coords):
    scaled_coords = (input_coords + 23.25) * 10
    return scaled_coords


def scale_coords_leoben(input_coords):
    scaled_coords = input_coords * 10
    return scaled_coords


def plot_wrapper(triangle_coords, scaled_coords, fillings, cache_dir, imsize, index):
    fillings = np.squeeze(fillings)
    filling = fillings[index]
    im_fn = cache_dir / f'{index}.png'
    load_image = True
    if cache_dir != '' and not im_fn.exists():
        try:
            a = filling[triangle_coords]
            b = a.reshape(len(triangle_coords), 3)
            means_of_neighbour_nodes = b.mean(axis=1)
        except IndexError:
            print('ERROR plot wrapper ... raising')
            print(triangle_coords)
            print(filling[triangle_coords])
            raise

        im = draw_polygon_map(means_of_neighbour_nodes, scaled_coords, triangle_coords, colored=False)
        # im = create_np_image((465,465), scaled_coords, filling)
        # im_t = Image.fromarray(im,mode='L')
        im.save(im_fn)
        load_image = False
    else:
        try:
            im = Image.open(im_fn)
        except IndexError:
            print('ERROR: Corrupt img data')
            raise
    if im.size != imsize:
        im = im.resize(imsize)
    dat = np.asarray(im)
    im.close()
    return dat, index, load_image


def draw_polygon_map(values_for_triangles, scaled_coords, triangle_coords, colored=False, size=(465, 465)):
    mode = 'RGB' if colored else 'L'
    im = Image.new(mode, size)
    draw = ImageDraw.Draw(im)
    for i, triangle_coord in enumerate(triangle_coords):
        val = values_for_triangles[i]
        if not colored:
            pol = scaled_coords[triangle_coords[i - 1]]
            draw.polygon(pol, fill=(int(val * 255)))
        else:
            if val == 0.0:
                draw.polygon(scaled_coords[triangle_coord], fill=(255, 0, 0))
            elif val == 1.0:
                draw.polygon(scaled_coords[triangle_coord], fill=(0, 102, 255))
            else:
                h = 3.6 * val
                col = tuple(int(round(i * 255)) for i in colorsys.hsv_to_rgb(h, 1, 1))
                draw.polygon(scaled_coords[triangle_coord], fill=col)
    return im
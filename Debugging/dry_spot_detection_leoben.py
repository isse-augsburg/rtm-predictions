import socket
from pathlib import Path
from time import time

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from PIL import Image
from PIL import ImageDraw


def scale_coords_leoben(input_coords):
    scaled_coords = input_coords * 10
    return scaled_coords

def draw_polygon_map(values_for_triangles, scaled_coords, triangle_coords,
                     colored=False, size=(465, 465)):
    mode = 'RGB' if colored else 'L'
    im = Image.new(mode, (np.max(scaled_coords[:,0]).astype(int), np.max(scaled_coords[:,1].astype(int))))
    draw = ImageDraw.Draw(im)
    for i, triangle_coord in enumerate(triangle_coords):
        val = values_for_triangles[i]
        pol = scaled_coords[triangle_coords[i - 1]]
        draw.polygon(pol, fill=(int(val * 255)))
    return im


def create_local_properties_map(
        data, scaled_coords, triangle_coords, _type="FIBER_FRACTION"
):
    values_for_triangles = data[_type]["ZONE1_set1"]["erfblock"]["res"][()]
    im = draw_polygon_map(values_for_triangles, scaled_coords, triangle_coords, size=(152 * 3, 120 * 3))
    return im

def create_permeability_map(f, imsize):
    coord_as_np_array = f[
        "post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"
    ][()]
    _all_coords = coord_as_np_array[:, :-1]
    
    scaled_coords = scale_coords_leoben(_all_coords)
    # norm_cords = normalize_coords(_all_coords)
    triangle_coords = f["post/constant/connectivities/SHELL/erfblock/ic"][()]
    triangle_coords = triangle_coords[:, :-1] - 151980 # required for Leoben data

    data = f["post/constant/entityresults/SHELL/"]

    im = create_local_properties_map(
        data, scaled_coords, triangle_coords, "FIBER_FRACTION"
    )
    if im.size != imsize:
        im = im.resize(imsize)
    return im, scaled_coords, triangle_coords



def analyze_image(img):
    _, threshold = cv2.threshold(img, 70, 190, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_COMPLEX
    triangles, rectangles, pentagons, ellipses, circles = [], [], [], [], []
    min_size = 3
   
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
        (x,y,w,h) = cv2.boundingRect(approx)
        #cv2.rectangle(img, (x,y), (x+w,y+h), 100, 2)
       
        size = cv2.contourArea(cnt)
        if size < min_size:
            continue
        # max countour
        if size > 273440:
            continue
       
        tmp = np.squeeze(approx)
        cv2.fillPoly(img, [np.squeeze(approx)], 130)
      
        
        # cv2.drawContours(img, [approx], 0, 175, 3)
        xx = approx.ravel()[0]
        yy = approx.ravel()[1]
        if len(approx) == 3:
            cv2.putText(img, f"Triangle: {size}", (xx, yy), font, 0.3, 0)
            triangles.append(size)
        elif len(approx) == 4:
            cv2.putText(img, f"Rectangle: {size}", (xx, yy), font, 0.3, 0)
            rectangles.append(size)
        elif len(approx) == 5:
            cv2.putText(img, f"Pentagon: {size}", (xx, yy), font, 0.3, 0)
            pentagons.append(size)
        elif 6 < len(approx) < 15:
            cv2.putText(img, f"Ellipse: {size}", (xx, yy), font, 0.3, 0)
            ellipses.append(size)
        else:
            cv2.putText(img, f"Circle: {size}", (xx, yy), font, 0.3, 0)
            circles.append(size)
   
    return triangles, rectangles, pentagons, ellipses, circles, img, threshold


def analyze_image_v2(img, perm_map=None):
    _, threshold = cv2.threshold(img, 70, 190, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_COMPLEX
    triangles, rectangles, pentagons, ellipses, circles = [], [], [], [], []
    min_size = 3
    dryspots = np.zeros_like(img, dtype=np.float)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
        (x,y,w,h) = cv2.boundingRect(approx)
        #cv2.rectangle(img, (x,y), (x+w,y+h), 100, 2)
       
        size = cv2.contourArea(cnt)
        if size < min_size:
            continue
        # max countour
        if size > 273440:
            continue
       
        empty = np.zeros_like(img, dtype=np.float)
        cv2.fillPoly(empty, [np.squeeze(approx)], 255)
        perm_cut = np.where(empty == 255, perm_map.astype(np.float), 0) # take values from perm map, where contour is
        perm_cut = np.where((perm_cut <= 70) & (perm_cut >= 65), 0, perm_cut) #filter values of the background assuming background is between 70 and 65
        perm_cut = np.where((perm_cut == 0), 0 , 255) # focus on anything other than background
        avg_dryspot_prob = np.sum(perm_cut,dtype=np.float) / size # normalize with size of contour area
        print(avg_dryspot_prob, np.sum(perm_cut,dtype=np.float), size) # debug print statement
        if avg_dryspot_prob > 220:
            cv2.fillPoly(dryspots, [np.squeeze(approx)], 255)
        
        # cv2.drawContours(img, [approx], 0, 175, 3)
        xx = approx.ravel()[0]
        yy = approx.ravel()[1]
        if len(approx) == 3:
            cv2.putText(img, f"Triangle: {size}", (xx, yy), font, 0.3, 0)
            triangles.append(size)
        elif len(approx) == 4:
            cv2.putText(img, f"Rectangle: {size}", (xx, yy), font, 0.3, 0)
            rectangles.append(size)
        elif len(approx) == 5:
            cv2.putText(img, f"Pentagon: {size}", (xx, yy), font, 0.3, 0)
            pentagons.append(size)
        elif 6 < len(approx) < 15:
            cv2.putText(img, f"Ellipse: {size}", (xx, yy), font, 0.3, 0)
            ellipses.append(size)
        else:
            cv2.putText(img, f"Circle: {size}", (xx, yy), font, 0.3, 0)
            circles.append(size)
   
    return triangles, rectangles, pentagons, ellipses, circles, img, threshold, dryspots


def analyze_for_path(path, id):
    f = h5py.File(path, 'r')
    im, scaled_coords, triangle_coords = create_permeability_map(f, (573, 420))
    #im = cv2.medianBlur(np.array(im),13)
    # triangles, rectangles, pentagons, ellipses, circles, im2, threshold = analyze_image(np.array(im))
    if socket.gethostname() == 'swtse130':
        cv2.imwrite(str(r'C:\Users\stiebesi\Desktop\shapes\test\%d_perm.png' % id), np.flip(np.array(im), 0))
    else:
        cv2.imwrite(str('/home/schroeter/Desktop/folder/test%d_perm.png' % (id)), np.flip(np.array(im), 0))
    # cv2.imwrite(str('/home/schroeter/Desktop/folder/test%d_perm.png' % (id)), np.flip(np.array(im),0))
    #cv2.imwrite(str('/home/schroeter/Desktop/folder/test%d_shapes_im.png' % ( id)), np.flip(np.array(im),0))
    return np.flip(np.array(im), 0)


def dry_spot_analysis(path, id, perm_map=None):
    f = h5py.File(path,'r')

    t00 = time()
    use_orig_mesh = True

    coord_as_np_array = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"][()]
    output_dir = Path(r'C:\Users\stiebesi\Desktop\shapes\shape_detection%d' % id)
    output_dir.mkdir(exist_ok=True, parents=True)
    _all_coords = coord_as_np_array[:, :-1]
    scaled_coords = scale_coords_leoben(_all_coords)
    x = scaled_coords[:, 0]
    y = scaled_coords[:, 1]
    # index_min_x = np.argmin(x)
    # index_max_x = np.argmax(x)
    # index_min_y = np.argmin(y)
    # index_max_y = np.argmin(y)
    # right_edge = np.where(x == 375)[0]
    # left_edge = np.where(x == 0)[0]
    # upper_edge = np.where(y == 300)[0]
    # lower_edge = np.where(y == 0)[0]
    #
    # upper_left = np.append(upper_edge[0:3], left_edge[0:3])
    # upper_right = np.append(upper_edge[-3:], right_edge[0:3])
    # lower_left = np.append(lower_edge[0:3], left_edge[-3:])
    # lower_right = np.append(lower_edge[-3:], left_edge[-3:])

    if use_orig_mesh:
        triangles = f['/post/constant/connectivities/SHELL/erfblock/ic'][()]
        triangles = triangles - triangles.min()
        triangles = triangles[:, :-1]
        xi = np.linspace(0, 375, 376)
        yi = np.linspace(0, 300, 301)
        Xi, Yi = np.meshgrid(xi, yi)
        triang = tri.Triangulation(x, y, triangles=triangles)

    keys = list(f['/post/singlestate'].keys())

    start_detecting_changes_at = 5
    changes, ups, downs = [], [], []
    num_shapes = 0

    # Fiber fraction map creation with tripcolor
    fvc = f["/post/constant/entityresults/SHELL/FIBER_FRACTION/ZONE1_set1/erfblock/res"][()].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tripcolor(triang, fvc, cmap='gray')

    ax.set(xlim=(0, 375), ylim=(0, 300))
    plt.axis('off')
    plt.tight_layout()

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.clim(0, 1)
    plt.savefig(r'C:\Users\stiebesi\Desktop\shapes\test_tripcolor\%d_fvc.png' % id, bbox_inches=extent)
    plt.close()

    #keys = [keys[111], keys[112], keys[113], keys[114], keys[115], keys[116], keys[117], keys[118], keys[119]]

    for i, k in enumerate(keys):
        data = f[f"/post/singlestate/{k}/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res"][()]
        z = data.flatten()

        fig = plt.figure()
        ax2 = fig.add_subplot(111)

        if use_orig_mesh:
            interpolator = tri.LinearTriInterpolator(triang, z)
            zi = interpolator(Xi, Yi)
            ax2.contourf(xi, yi, zi, levels=10, cmap='gray', extend='both')
            # ax2.contourf(xi, yi, zi, levels=20, colors='k')
            # ax2.contourf(xi, yi, zi, levels=20, cmap="RdBu_r")
        else:
            ax2.tricontourf(x, y, z, levels=10, colors='k')
        ax2.set(xlim=(0, 375), ylim=(0, 300))
        plt.axis('off')
        plt.tight_layout()

        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        plt.savefig(output_dir / ('test%d.png' % (i + 1)), bbox_inches=extent)
        plt.close()

        img = cv2.imread(str(output_dir / ('test%d.png' % (i + 1))), cv2.IMREAD_GRAYSCALE)
        perm_map = cv2.imread(r'C:\Users\stiebesi\Desktop\shapes\test_tripcolor\%d_fvc.png' % id, cv2.IMREAD_GRAYSCALE)
        # add a border
        # img = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)
        img = 255 - img
        _triangles, rectangles, pentagons, ellipses, circles, img, threshold, dryspots = analyze_image_v2(img, perm_map)
        cv2.imwrite(str(output_dir / ('test%d_dry.png' % (i + 1))), dryspots)
        # cv2.imwrite(str(output_dir / ('test%d_threshold.png' % (i + 1))), threshold)
        all = len(_triangles) + len(rectangles) + len(pentagons) + len(ellipses) + len(circles)
        if all > num_shapes:
            num_shapes = all
            if i >= start_detecting_changes_at:
                changes.append(i + 1)
                ups.append(i + 1)
                print('Up detected!')
        if all < num_shapes:
            num_shapes = all
            if i >= start_detecting_changes_at:
                changes.append(i + 1)
                downs.append(i + 1)
                print('Down detected!')
        print(
            f"{i} Found: all: {all}, tri: {_triangles}, rect: {rectangles}, pent: {pentagons}, ell: {ellipses}, circ: {circles}")
    print(
        f'{id} Overall time: {time() - t00}. Remember: Indeces in PAM RTM start with 1, images have pam-like indices. '
        f'Changes detected: {changes}, Ups: {ups}, Downs: {downs}, delta: {len(ups) - len(downs)}')


if __name__ == "__main__":
    if socket.gethostname() == "swtse130":
        source = Path(r'X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-07-23_15-38-08_5000p')
    else:
        source = Path('/home/schroeter/Desktop/')
    # for i in range(5, 11):
    i = 7
    perm0 = analyze_for_path(source / str(i) / str('2019-07-23_15-38-08_%d_RESULT.erfh5' % i), i)
    dry_spot_analysis(source / str(i) / str('2019-07-23_15-38-08_%d_RESULT.erfh5' % i), i, perm0)

    # perm1 = analyze_for_path(source / '7' / '2019-07-23_15-38-08_7_RESULT.erfh5', 1)
    # perm2 = analyze_for_path(source / '3' / '2019-08-24_11-51-48_3_RESULT.erfh5', 2)
    # dry_spot_analysis(source / '7' / '2019-07-23_15-38-08_7_RESULT.erfh5', 7, perm1)
    # dry_spot_analysis(source / '3' / '2019-08-24_11-51-48_3_RESULT.erfh5', 2, perm2)
   

    # get_sensordata_and_flowfront(
    #     Path(r"/home/schroeter/Desktop/2019-08-24_11-51-48_3_RESULT.erfh5")
    # )
   
   
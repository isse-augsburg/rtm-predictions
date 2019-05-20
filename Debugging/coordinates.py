import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os import walk
from PIL import Image

def get_filelist_within_folder(root_directory):
    dataset_filenames = []
    for dirs in root_directory:

        for (dirpath, _, filenames) in walk(dirs):
            if filenames:
                filenames = [dirpath + '/' + f for f in filenames if f.endswith('.erfh5')]
                dataset_filenames.extend(filenames)
                print(len(dataset_filenames))

    return dataset_filenames

def get_coordinates_of_rectangle(filename, lower_left, height, width):
    """ lower_left =  [-5, -8]
    height = 15
    width = 0.125 """
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_rectangles = []

    corner_x = lower_left[0]
    corner_y = lower_left[1]

    current_rect = []
    for i in np.arange(lower_left[0], lower_left[0] + width, 0.125):
        for j in np.arange(lower_left[1], lower_left[1] + height, 0.125):
            index = np.where((_all_coords[:, 0] == [i]) & (_all_coords[:, 1] == [j]))
            index = index[0]
            if index.size != 0:
                current_rect.append(index)
    indices_of_rectangles.append(current_rect)

    x_coords = _all_coords[:, 0]
    y_coords = _all_coords[:, 1]

    plt.scatter(x_coords, y_coords)
    x_coords = np.take(x_coords, indices_of_rectangles)
    y_coords = np.take(y_coords, indices_of_rectangles)
    plt.scatter(x_coords, y_coords)

    plt.show()

    return indices_of_rectangles


def get_coordinates_of_circle(filename, circles):
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_circles = []

    x_coords = _all_coords[:, 0]
    y_coords = _all_coords[:, 1]

    for centre, radius in circles:
        current_indices = []
        for i in np.arange(centre[0] - radius, centre[0] + radius, 0.125):
            for j in np.arange(centre[1] - radius, centre[1] + radius, 0.125):
                distance = (i - centre[0]) ** 2 + (j - centre[1]) ** 2
                if distance <= radius ** 2:
                    index = np.where((_all_coords[:, 0] == [i]) & (_all_coords[:, 1] == [j]))
                    index = index[0]
                    if index.size != 0:
                        current_indices.append(index)
        indices_of_circles.append(current_indices)

    # plt.plot(x_coords, y_coords)

    # list that contains lists of the indices of circles
    return indices_of_circles


def get_elements_in_shape(filename, shape):
    f = h5py.File(filename, 'r')
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1]
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_elements = []

    for i in shape:
        current_elements = []
        for index, t in enumerate(triangle_coords):
            if t[0] in i and t[1] in i and t[2] in i:
                current_elements.append(index)
        indices_of_elements.append(current_elements)
    return indices_of_elements


def get_indices_of_elements_in_circles(filename, circles=(([-5, -5], 1.25), ([5, 5], 2), ([7, 0], 0.5))):
    f = h5py.File(filename, 'r')
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1]
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_circles = get_coordinates_of_circle(filename, circles)

    indices_of_elements = get_elements_in_shape(filename, indices_of_circles)

    for i in indices_of_elements:
        for x, y, z in _all_coords[triangle_coords[i]]:
            stack = np.stack([x, y, z, x])
            x_plot, y_plot = stack[:, 0], stack[:, 1]
            plt.plot(x_plot, y_plot)
        plt.show()

    return indices_of_elements


def get_indices_of_elements_in_rectangle(filename, lower_left=[-5, -8], height=3, width=0.5):
    f = h5py.File(filename, 'r')
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1]
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_rectangle = get_coordinates_of_rectangle(filename, lower_left, height, width)
    indices_of_elements = get_elements_in_shape(filename, indices_of_rectangle)

    for i in indices_of_elements:
        for x, y, z in _all_coords[triangle_coords[i]]:
            stack = np.stack([x, y, z, x])
            x_plot, y_plot = stack[:, 0], stack[:, 1]
            plt.plot(x_plot, y_plot)
        plt.show()

    return indices_of_elements


def plot_weird_coordinates(filename):
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1]
    test_instance = triangle_coords[::10] - 1

    for x, y, z in _all_coords[test_instance]:
        stack = np.stack([x, y, z, x])
        x_plot, y_plot = stack[:, 0], stack[:, 1]
        plt.plot(x_plot, y_plot)
    plt.show()

    return


def perm_map(filename):
    f = h5py.File(filename, 'r')
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    perm1 = f['post/constant/entityresults/SHELL/PERMEABILITY1/ZONE1_set1/erfblock/res'].value
    perm2 = f['post/constant/entityresults/SHELL/PERMEABILITY2/ZONE1_set1/erfblock/res'].value
    perm3 = f['post/constant/entityresults/SHELL/PERMEABILITY3/ZONE1_set1/erfblock/res'].value
    fvc = f['post/constant/entityresults/SHELL/FIBER_FRACTION/ZONE1_set1/erfblock/res'].value
    perm1 = perm1[:,0]
    perm2 = perm2[:,1]
    perm3 = perm3[:,2]
    perm = np.stack((perm1,perm2,perm3), axis=-1)
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1] -1
    _all = _all_coords[triangle_coords]
   
    _meaned = np.mean(_all, axis = 1);
    _meaned= normalize(_meaned);
    fvc = normalize(fvc)
    #perm = normalize(perm)
    create_np_image(norm_coords=_meaned, data = fvc)
    
def normalize(coords):
    max_c = np.max(coords)
    min_c = np.min(coords)
    coords = coords - min_c
    coords = coords / (max_c - min_c)
    return coords

def create_np_image(target_shape=(1000, 1000, 1), norm_coords=None, data=None ):
    if norm_coords is None or data is None:
        print("ERROR")
        return
    assert np.shape(norm_coords)[0] == np.shape(data)[0]

    arr = np.zeros(target_shape)

   
    # np.expand_dims(data, axis = 1)
    norm_coords[:, 0] = norm_coords[:, 0] * (target_shape[0] - 1)
    norm_coords[:, 1] = norm_coords[:, 1] * (target_shape[1] - 1)
    data = data * 255
    norm_coords = norm_coords.astype(np.int)
    arr[norm_coords[:, 0], norm_coords[:, 1]] = data
    arr= np.squeeze(arr)
    img = Image.fromarray(arr, 'L')
    img.save('my.png')
    img.show()

    return arr  
    


if __name__ == "__main__":
    list2 = get_filelist_within_folder(['/run/user/1002/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-04-26_15-09-31_100p_20_shapes'])
    # get_coordinates_of_rectangle('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
    for el in list2:
        perm_map(el)
    # plot_weird_coordinates('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')

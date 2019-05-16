import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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


if __name__ == "__main__":
    # get_coordinates_of_rectangle('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
    #get_indices_of_elements_in_rectangle('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
    # plot_weird_coordinates('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
    coords = get_indices_of_elements_in_rectangle('Y:/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
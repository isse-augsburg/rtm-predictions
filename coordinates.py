import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib


def get_coordinates_of_rectangle(filename):
    lower_left =  [-5, -8]
    height = 15
    width = 0.125
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_rectangle = list()

    corner_x = lower_left[0]
    corner_y = lower_left[1]

    for i in np.arange(lower_left[0], lower_left[0]+width, 0.125):
        for j in np.arange(lower_left[1], lower_left[1]+height, 0.125):
            index = np.where((_all_coords[:,0] == [i]) & (_all_coords[:,1] == [j]))
            index = index[0]
            if index.size != 0:
                indices_of_rectangle.append(index)
    
    x_coords = _all_coords[:,0]
    y_coords = _all_coords[:,1]
    

    plt.scatter(x_coords, y_coords)
    x_coords = np.take(x_coords, indices_of_rectangle)
    y_coords = np.take(y_coords, indices_of_rectangle)
    plt.scatter(x_coords, y_coords)

    plt.show()

    return indices_of_rectangle



if __name__ == "__main__":
    get_coordinates_of_rectangle('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
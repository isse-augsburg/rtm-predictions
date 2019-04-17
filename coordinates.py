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

def get_coordinates_of_circle(filename, circles=(([-5, -5], 1.25), ([5, 5], 0.25), ([7, 0], 0.5))):
    
    

    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_circles = list()

    x_coords = _all_coords[:,0]
    y_coords = _all_coords[:,1]

    for centre, radius in circles:
        current_indices = list()
        for i in np.arange(centre[0]-radius, centre[0]+radius, 0.125):
            for j in np.arange(centre[1]-radius, centre[1]+radius, 0.125):
                distance = (i - centre[0])**2 + (j-centre[1])**2 
                if distance <= radius**2:
                    index = np.where((_all_coords[:,0] == [i]) & (_all_coords[:,1] == [j]))
                    index = index[0]
                    if index.size != 0:
                        current_indices.append(index)
        indices_of_circles.append(current_indices)
    
    plt.plot(x_coords, y_coords)

    for i in indices_of_circles:
        x_plot = np.take(x_coords, i)
        y_plot = np.take(y_coords, i)
        plt.plot(x_plot, y_plot, 'r.')
    plt.show()

    #list that contains lists of the indices of circles
    return indices_of_circles



if __name__ == "__main__":
    #get_coordinates_of_rectangle('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
    get_coordinates_of_circle('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import random 
import time 

def get_coordinates_of_rectangle(filename, sizes):
    """ lower_left =  [-5, -8]
    height = 15
    width = 0.125 """
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_rectangles = []

    current_rect = []
    for s in sizes:
        lower_left = s[0]
        width = s[1]
        height = s[2]

        for i in np.arange(lower_left[0], lower_left[0]+width, 0.125):
            for j in np.arange(lower_left[1], lower_left[1]+height, 0.125):
                index = np.where((_all_coords[:,0] == [i]) & (_all_coords[:,1] == [j]))
                index = index[0]
                if index.size != 0:
                    current_rect.append(int(index))
        indices_of_rectangles.append(set(current_rect))
    
    x_coords = _all_coords[:,0]
    y_coords = _all_coords[:,1]
    

    return indices_of_rectangles

def get_coordinates_of_circle(filename, circles):
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_circles = []

    x_coords = _all_coords[:,0]
    y_coords = _all_coords[:,1]

    for centre, radius in circles:
        current_indices = []
        for i in np.arange(centre[0]-radius, centre[0]+radius, 0.125):
            for j in np.arange(centre[1]-radius, centre[1]+radius, 0.125):
                distance = (i - centre[0])**2 + (j-centre[1])**2 
                if distance <= radius**2:
                    index = np.where((_all_coords[:,0] == [i]) & (_all_coords[:,1] == [j]))
                    index = index[0]
                    if index.size != 0:
                        current_indices.append(int(index))
        indices_of_circles.append(set(current_indices))

    #list that contains lists of the indices of circles
    return indices_of_circles
def get_elements_in_shape(filename, shape):
    f = h5py.File(filename, 'r')
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()]
    triangle_coords = triangle_coords[:, :-1]
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
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

def get_indices_of_elements_in_circles(filename, circles=(([-24, -5], 5), ([5, 5], 2), ([7, 0], 0.5))):
    f = h5py.File(filename, 'r')
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()]
    triangle_coords = triangle_coords[:, :-1]
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]
    
    indices_of_circles = get_coordinates_of_circle(filename, circles)

    indices_of_elements = get_elements_in_shape(filename, indices_of_circles)
    
    

    return indices_of_elements

    

def get_indices_of_elements_in_rectangle(filename, sizes=[((-5, -8), 3, 0.5), ((5, 0), 1, 1)] ):
    f = h5py.File(filename, 'r')
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()]
    triangle_coords = triangle_coords[:, :-1]
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_rectangle = get_coordinates_of_rectangle(filename, sizes)
    indices_of_elements = get_elements_in_shape(filename, indices_of_rectangle)
    
    return indices_of_elements

def get_coordinates_of_runner(filename, sizes): 
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_runner = []

    current_runner = []
    for s in sizes:
        lower_left = s[0]
        width = s[1]
        height = s[2]
        runner_lower_left = s[3]
        runner_width = s[4]
        runner_height = s[5]

        for i in np.arange(lower_left[0], lower_left[0]+width, 0.125):
            for j in np.arange(lower_left[1], lower_left[1]+height, 0.125):
                if((i > runner_lower_left[0] and i < runner_lower_left[0]+runner_width) and  
                (j > runner_lower_left[1] and j < runner_lower_left[1] + runner_height)):
                    continue

                index = np.where((_all_coords[:,0] == [i]) & (_all_coords[:,1] == [j]))
                index = index[0]
                if index.size != 0:
                    current_runner.append(int(index))
        indices_of_runner.append(set(current_runner))
    
    x_coords = _all_coords[:,0]
    y_coords = _all_coords[:,1]
    

    return indices_of_runner



#sizes = [((lower_left_x, lower_left_y), width, height, (runner_lower_left_x, runner_lower_left_y), runner_width, runner_height)]
def get_indices_of_elements_in_runner(filename, sizes=[((-5, -8), 3, 10, (-4, -6), 1, 20)]):
    f = h5py.File(filename, 'r')
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()]
    triangle_coords = triangle_coords[:, :-1]
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    indices_of_runner = get_coordinates_of_runner(filename, sizes)
    indices_of_elements = get_elements_in_shape(filename, indices_of_runner)
    return indices_of_elements





def plot_weird_coordinates(filename):
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'].value
    triangle_coords = triangle_coords[:, :-1]
    test_instance = triangle_coords[::10] - 1 



    for x,y,z in _all_coords[test_instance]:
        stack = np.stack([x, y, z, x])
        x_plot, y_plot = stack[:,0], stack[:,1]
        plt.plot(x_plot, y_plot)
    plt.show()   


    return

def plot_triangles(filename, shapes):
    f = h5py.File(filename, 'r')
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()]
    triangle_coords = triangle_coords[:, :-1]
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
    # Cut off last column (z), since it is filled with 1s anyway
    _all_coords = coord_as_np_array[:, :-1]

    for i in shapes: 
        for x, y, z in _all_coords[triangle_coords[i]]:
            stack = np.stack([x, y, z, x])
            x_plot, y_plot = stack[:,0], stack[:,1]
            plt.plot(x_plot, y_plot)
    plt.show()

def get_rounded_random(value, minClip):
    return round(float(value)/ minClip) * minClip

def rounded_normal_distributed(mu, sigma, minClip): 
    value = np.random.normal(mu, sigma)
    return abs(get_rounded_random(value, minClip))



def get_elements_of_random_shapes(filename):
    num_rectangles = 5
    num_circles = 7

    rectangle_sizes = []
    circle_sizes = []

    neg_x_border = -24
    pos_x_border = 24 
    neg_y_border = -24
    pos_y_border = 24

    min_side_length = 0.25
    max_side_length = 4

    mu_side_length = 2.5
    sigma_side_length = 1.5

    mu_radius = 1.0
    sigma_radius = 1.5

    min_radius = 0.25
    max_radius = 3

    for i in range(num_rectangles):
        
        height = rounded_normal_distributed(mu_side_length, sigma_side_length, 0.125)
        width = rounded_normal_distributed(mu_side_length, sigma_side_length, 0.125)
        lower_left_x = get_rounded_random(random.random() * (pos_x_border - neg_x_border - width) + neg_x_border, 0.125)
        lower_left_y = get_rounded_random(random.random() * (pos_y_border - neg_y_border - height) + neg_y_border, 0.125)
        rectangle_sizes.append(((lower_left_x, lower_left_y), height, width))
    
    rectangle_sizes = get_indices_of_elements_in_rectangle(filename, rectangle_sizes)


    for i in range(num_circles):
        radius = rounded_normal_distributed(mu_radius, sigma_radius, 0.125)
        center_x = get_rounded_random(random.random() * (pos_x_border - neg_x_border - 2*radius) + neg_x_border + radius, 0.125)
        center_y = get_rounded_random(random.random() * (pos_y_border - neg_y_border - 2*radius) + neg_y_border + radius, 0.125)
        circle_sizes.append(((center_x, center_y), radius))
    
    circle_sizes = get_indices_of_elements_in_circles(filename, circle_sizes)
    
    return rectangle_sizes, circle_sizes


    


if __name__ == "__main__":
    path = 'Y:/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5'
    #path = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5'
    #shapes = get_indices_of_elements_in_rectangle(path)
    #shapes = get_indices_of_elements_in_circles(path)
    #plot_weird_coordinates('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/2019-04-01_14-02-51_k1_pertubated_sigma1.110e-11_mu0.0_369_RESULT.erfh5')
    #rects, circles = get_elements_of_random_shapes(path)
    #rects.extend(circles)
    runners = get_indices_of_elements_in_runner(path)
    plot_triangles(path, runners)
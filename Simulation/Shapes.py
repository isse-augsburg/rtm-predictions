import h5py
import numpy as np
import random
from Simulation.helper_functions import *
import Simulation.resources as resources

class Rectangle:
    def __init__(self, lower_left=(0,0), height=0, width=0):
        self.lower_left = lower_left
        self.height = height
        self.width = width

class Circle:
    def __init__(self, center=(0,0), radius=0):
        self.center = center
        self.radius = radius

class Runner: 
    def __init__(self, lower_left=(0,0), height=0, width=0):
        self.lower_left = lower_left
        self.height = height
        self.width = width

class Shaper:
    def __init__(self, reference_erfh5, perturbation_factors):
        self.perturbation_factors = perturbation_factors
        f = h5py.File(reference_erfh5, 'r')
        self.all_coords= f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()][:, :-1]
        self.triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()][:, :-1]
        self.x_bounds = (-20, 20)
        self.y_bounds = (-20, 20)
        self.circ_radius_bounds = (1, 3)
        self.rect_width_bounds = self.rect_height_bounds = (1, 8)
        self.grid_step = 0.125
        self.shapes = []
        if len(self.perturbation_factors.keys()) > 0:
            [self.shapes.append(Rectangle) for x in range(self.perturbation_factors['Shapes']['Rectangles']['Num'])]
            [self.shapes.append(Circle) for x in range(self.perturbation_factors['Shapes']['Circles']['Num'])]
            [self.shapes.append(Runner) for x in range(self.perturbation_factors['Shapes']['Runners']['Num'])]
            self.rect_fvc_bounds = self.perturbation_factors['Shapes']['Rectangles']['Fiber_Content']
            self.circ_fvc_bounds = self.perturbation_factors['Shapes']['Circles']['Fiber_Content']
            self.runner_fvc_bounds = self.perturbation_factors['Shapes']['Runners']['Fiber_Content']

    def place_runner(self, _dict):
        fvc = random.random() * (self.runner_fvc_bounds[1] - self.runner_fvc_bounds[0]) + self.runner_fvc_bounds[0]
        lower_left_x, lower_left_y = -5, -8
        overall_width, overall_height = 3, 10
        runner_x, runner_y = -4, -6
        runner_width, runner_height = 1, 6
        print("Creating runner")
        set_of_indices_of_shape = self.get_coordinates_of_runner(((lower_left_x, lower_left_y), overall_width,
                                                                   overall_height, (runner_x, runner_y), runner_width,
                                                                   runner_height))
        print(set_of_indices_of_shape)
        _dict = {"Runner":
                     {"fvc": fvc,
                      "height": overall_height,
                      "width": overall_width,
                      "inner_height": runner_height,
                      "inner_width": runner_width
                      }
                 }
        return _dict, fvc, set_of_indices_of_shape

    def place_circle(self, x, y):
        fvc = random.random() * (self.circ_fvc_bounds[1] - self.circ_fvc_bounds[0]) + self.circ_fvc_bounds[0]
        radius = rounded_random(
            random.random() * (self.circ_radius_bounds[1] - self.circ_radius_bounds[0]) + self.circ_radius_bounds[0],
            self.grid_step)
        set_of_indices_of_shape = self.get_coordinates_of_circle((x, y), radius)
        _dict = {"Circle":
                     {"fvc": fvc,
                      "radius": radius,
                      }
                 }
        return _dict, fvc, set_of_indices_of_shape

    def place_rectangle(self, x, y):
        fvc = random.random() * (self.rect_fvc_bounds[1] - self.rect_fvc_bounds[0]) + self.rect_fvc_bounds[0]
        height = rounded_random(
            random.random() * (self.rect_height_bounds[1] - self.rect_height_bounds[0]) + self.rect_height_bounds[0],
            self.grid_step)
        width = rounded_random(
            random.random() * (self.rect_width_bounds[1] - self.rect_width_bounds[0]) + self.rect_width_bounds[0],
            self.grid_step)
        set_of_indices_of_shape = self.get_coordinates_of_rectangle((x, y), height, width)
        _dict = {"Rectangle":
                     {"fvc": fvc,
                      "height": height,
                      "width": width
                      }
                 }
        return _dict, fvc, set_of_indices_of_shape

    def get_coordinates_of_runner(self, s):
        current_runner = set()
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

                index = np.where((self.all_coords[:,0] == [i]) & (self.all_coords[:,1] == [j]))
                index = index[0]
                if index.size != 0:
                    current_runner.add(int(index))

        return current_runner

    def get_coordinates_of_rectangle(self, lower_left, height, width):
        current_rect = set()

        for i in np.arange(lower_left[0], lower_left[0] + width, self.grid_step):
            for j in np.arange(lower_left[1], lower_left[1] + height, self.grid_step):
                index = np.where((self.all_coords[:, 0] == [i]) & (self.all_coords[:, 1] == [j]))[0]
                if index.size != 0:
                    current_rect.add(index[0])
        return current_rect

    def get_coordinates_of_circle(self, centre, radius):
        current_indices = set()
        for i in np.arange(centre[0]-radius, centre[0]+radius, self.grid_step):
            for j in np.arange(centre[1]-radius, centre[1]+radius, self.grid_step):
                distance = (i - centre[0])**2 + (j-centre[1])**2
                if distance <= radius**2:
                    index = np.where((self.all_coords[:,0] == [i]) & (self.all_coords[:,1] == [j]))
                    index = index[0]
                    if index.size != 0:
                        current_indices.add(index[0])

        return current_indices

    def get_elements_in_shape(self, indeces_nodes):
        current_elements = list()
        for index, t in enumerate(self.triangle_coords):
            if t[0] in indeces_nodes and t[1] in indeces_nodes and t[2] in indeces_nodes:
                current_elements.append(index)
        return current_elements

    def apply_shapes(self, df, save_to_h5_data):
        set_of_indices_of_shape = set()
        for i, shape in enumerate(self.shapes):
            y = rounded_random(random.random() * (self.y_bounds[1] - self.y_bounds[0]) + self.y_bounds[0],
                               self.grid_step)
            x = rounded_random(random.random() * (self.x_bounds[1] - self.x_bounds[0]) + self.x_bounds[0],
                               self.grid_step)
            if 'shapes' not in save_to_h5_data.keys():
                save_to_h5_data['shapes'] = []
            if shape.__name__ == 'Rectangle':
                _dict, fvc, set_of_indices_of_shape = self.place_rectangle(x, y)

            elif shape.__name__ == 'Circle':
                _dict, fvc, set_of_indices_of_shape = self.place_circle(x, y)

            elif shape.__name__ == 'Runner':
                # FIXME not yet working with x and y as parameters
                _dict, fvc, set_of_indices_of_shape = self.place_runner(_dict)

            save_to_h5_data['shapes'].append(_dict)

            indices_of_elements = self.get_elements_in_shape(set_of_indices_of_shape)
            df.update(df.iloc[indices_of_elements]['Fiber_Content'] * (1 + fvc))
        return save_to_h5_data
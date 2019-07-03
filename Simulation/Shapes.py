from enum import Enum

import h5py
import numpy as np
import random

from Pipeline.plots_and_images import draw_polygon_map, scale_coords_lautern
from Simulation.helper_functions import *
import Simulation.resources as resources


class Shape:
    def __init__(self, fvc=None):
        self.fvc = fvc


class Rectangle(Shape):
    def __init__(self, lower_left=(0, 0), height=0, width=0, fvc=None):
        super().__init__(fvc)
        self.lower_left = lower_left
        self.height = height
        self.width = width


class Circle(Shape):
    def __init__(self, center=(0, 0), radius=0, fvc=None):
        super().__init__(fvc)
        self.center = center
        self.radius = radius
        self.width = radius * 2
        self.height = radius * 2


class Runner(Shape):
    def __init__(self, lower_left=(0, 0), height=0, width=0, fvc=None):
        super().__init__(fvc)
        self.lower_left = lower_left
        self.height = height
        self.width = width

class TargetSimulation(Enum):
    Lautern = 0
    Leoben = 1
    KMT = 2

class Shaper:
    def __init__(self, reference_erfh5, perturbation_factors, grid_step=0.125, target=TargetSimulation.Lautern):
        self.perturbation_factors = perturbation_factors
        f = h5py.File(reference_erfh5, 'r')
        self.all_coords = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()][:, :-1]
        self.triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()][:, :-1]
        self.x_bounds = (self.all_coords[:, 0].min(), self.all_coords[:, 0].max())
        self.y_bounds = (self.all_coords[:, 1].min(), self.all_coords[:, 1].max())
        self.circ_radius_bounds = (1, 3)
        self.rect_width_bounds = self.rect_height_bounds = (1, 8)
        self.grid_step = grid_step
        # self.grid_step = 0.002377933
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

    def place_circle(self, x, y, radius=None):
        fvc = random.random() * (self.circ_fvc_bounds[1] - self.circ_fvc_bounds[0]) + self.circ_fvc_bounds[0]
        if radius is None:
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

    def place_rectangle(self, x=None, y=None, height=None, width=None, fvc=None):
        if x is None and y is None:
            x, y = self.get_random_coords_in_bounds()
        if fvc is None:
            fvc = random.random() * (self.rect_fvc_bounds[1] - self.rect_fvc_bounds[0]) + self.rect_fvc_bounds[0]
        if height is None and width is None:
            height, width = self.get_random_height_width()
        set_of_indices_of_shape = self.get_coordinates_of_rectangle((x, y), height, width)
        # if len(set_of_indices_of_shape):

        _dict = {"Rectangle":
                     {"fvc": fvc,
                      "height": height,
                      "width": width
                      }
                 }
        return _dict, fvc, set_of_indices_of_shape

    def get_random_height_width(self):
        height = rounded_random(
            random.random() * (self.rect_height_bounds[1] - self.rect_height_bounds[0]) + self.rect_height_bounds[0],
            self.grid_step)
        width = rounded_random(
            random.random() * (self.rect_width_bounds[1] - self.rect_width_bounds[0]) + self.rect_width_bounds[0],
            self.grid_step)
        return height, width

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

        # for i in np.arange(lower_left[0], lower_left[0] + width, self.grid_step):
        #     for j in np.arange(lower_left[1], lower_left[1] + height, self.grid_step):
        #         index = np.where((self.all_coords[:, 0] == [i]) & (self.all_coords[:, 1] == [j]))[0]
        #         if index.size != 0:
        #             current_rect.add(index[0])
        # return current_rect

        for i, point in enumerate(self.all_coords):
            if lower_left[0] < point[0] < lower_left[0] + width:
                if lower_left[1] < point[1] < lower_left[1] + height:
                    current_rect.add(i)
        return current_rect

    def get_coordinates_of_circle(self, centre, radius):
        current_circ = set()

        for i, point in enumerate(self.all_coords):
            if (point[0] - centre[0])**2 + (point[1] - centre[1])**2 <= radius**2:
                current_circ.add(i)

        # for i in np.arange(centre[0]-radius, centre[0]+radius, self.grid_step):
        #     for j in np.arange(centre[1]-radius, centre[1]+radius, self.grid_step):
        #         distance = (i - centre[0])**2 + (j-centre[1])**2
        #         if distance <= radius**2:
        #             index = np.where((self.all_coords[:,0] == [i]) & (self.all_coords[:,1] == [j]))
        #             index = index[0]
        #             if index.size != 0:
        #                 current_indices.add(index[0])

        return current_circ

    def get_elements_in_shape(self, indeces_nodes):
        current_elements = list()
        for index, t in enumerate(self.triangle_coords):
            if t[0] in indeces_nodes and t[1] in indeces_nodes and t[2] in indeces_nodes:
                current_elements.append(index)
        return current_elements

    def apply_shapes(self, df, save_to_h5_data, randomize=True):
        set_of_indices_of_shape = set()
        for i, shape in enumerate(self.shapes):
            if 'shapes' not in save_to_h5_data.keys():
                save_to_h5_data['shapes'] = []
            if shape.__class__.__name__ == 'Rectangle':
                if randomize:
                    _dict, fvc, set_of_indices_of_shape = self.place_rectangle()
                else:
                    x, y = shape.lower_left
                    _dict, fvc, set_of_indices_of_shape = self.place_rectangle(x, y, height=shape.height,
                                                                               width=shape.width, fvc=shape.fvc)
            elif shape.__class__.__name__ == 'Circle':
                if randomize:
                    x, y = self.get_random_coords_in_bounds()
                    _dict, fvc, set_of_indices_of_shape = self.place_circle(x, y)
                else:
                    x, y = shape.center
                    radius = shape.radius
                    _dict, fvc, set_of_indices_of_shape = self.place_circle(x, y, radius)
            elif shape.__class__.__name__ == 'Runner':
                # FIXME not yet working with x and y as parameters
                _dict, fvc, set_of_indices_of_shape = self.place_runner(_dict)

            save_to_h5_data['shapes'].append(_dict)

            indices_of_elements = self.get_elements_in_shape(set_of_indices_of_shape)
            df.update(df.iloc[indices_of_elements]['Fiber_Content'] * (1 + fvc))
        return save_to_h5_data

    def get_random_coords_in_bounds(self):
        x = rounded_random(random.random() * (self.x_bounds[1] - self.x_bounds[0]) + self.x_bounds[0],
                           self.grid_step)
        y = rounded_random(random.random() * (self.y_bounds[1] - self.y_bounds[0]) + self.y_bounds[0],
                           self.grid_step)
        return x, y

    def create_img_from_lperm(self, lperm_data):
        scaled_coords = scale_coords_lautern(self.all_coords)
        im = self.create_local_properties_map_lperm(lperm_data, scaled_coords, self.triangle_coords, 'Fiber_Content')
        return im, scaled_coords, self.triangle_coords

    @staticmethod
    def create_local_properties_map_lperm(data, scaled_coords, triangle_coords, _type='Fiber_Content'):
        values_for_triangles = data[_type]

        im = draw_polygon_map(values_for_triangles, scaled_coords, triangle_coords - triangle_coords.min(), size=(375, 300))
        return im

import logging
import cv2
import numpy as np


def scale_coords_lautern(input_coords):
    scaled_coords = (input_coords + 23.25) * 10
    return scaled_coords


def scale_coords_leoben(input_coords):
    scaled_coords = input_coords * 10
    return scaled_coords


def normalize_coords(coords):
    coords = np.array(coords)
    max_c = np.max(coords[:, 0])
    min_c = np.min(coords[:, 0])
    coords[:, 0] = coords[:, 0] - min_c
    coords[:, 0] = coords[:, 0] / (max_c - min_c)
    max_c = np.max(coords[:, 1])
    min_c = np.min(coords[:, 1])
    coords[:, 1] = coords[:, 1] - min_c
    coords[:, 1] = coords[:, 1] / (max_c - min_c)
    return coords

    # for new data 38 and 30.0


def create_np_image(target_shape=(143, 111), norm_coords=None, data=None):
    if norm_coords is None or data is None:
        logger = logging.getLogger(__name__)
        logger.error("ERROR in create_np_image")
        return

    assert np.shape(norm_coords)[0] == np.shape(data)[0]

    arr = np.zeros(target_shape)

    data = np.expand_dims(data, axis=1)
    coords_value = np.append(norm_coords, data, axis=1)
    coords_value[:, 0] = coords_value[:, 0] * (target_shape[0] - 1)
    coords_value[:, 1] = coords_value[:, 1] * (target_shape[1] - 1)
    coords_value[:, 2] = coords_value[:, 2]
    # coords_value = coords_value.astype(np.int)
    arr[coords_value[:, 0].astype(np.int), coords_value[:, 1].astype(np.int)] = coords_value[:, 2]

    return arr


def flip_array_diag(arr):
    arr2 = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.flip(arr2, 0)

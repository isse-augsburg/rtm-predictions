import numpy as np


def get_fixed_number_of_indices(list_length, num_samples):
    if num_samples >= list_length:
        return None

    step_size = list_length / num_samples

    # Starting np.arange in reversed order to include last step
    indices = np.array(np.arange(start=list_length-1, stop=0.0, step=-step_size))
    indices = np.around(indices)
    indices = np.flip(indices.astype(int))

    return indices


if __name__ == '__main__':
    i = get_fixed_number_of_indices(743, 77)
    print(i)

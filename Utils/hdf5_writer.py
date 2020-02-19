import h5py
import numpy as np


def create_hdf5(filename):
    return h5py.File(filename, 'a')


# outfile - a h5py file or group, dict_to_save - a dictionary
def write_dict_to_hdf5(outfile, dict_to_save: dict):
    for key, value in dict_to_save.items():
        if type(value) is dict:
            try:
                grp = outfile.create_group(str(key))
            except ValueError:
                grp = outfile[str(key)]

            write_dict_to_hdf5(grp, value)
        elif type(value) is list and type(value[0]) is dict:
            try:
                outfile = outfile.create_group(str(key))
            except ValueError:
                outfile = outfile[str(key)]
            wrapped = wrap_list(value)
            write_dict_to_hdf5(outfile, wrapped)
        else:
            outfile.create_dataset(str(key), data=np.array(value))


def wrap_list(inputlist):
    new_dict = {}
    for element in inputlist:
        dict_appendor(element, new_dict)
    return new_dict


def dict_appendor(element, dict_to_append):
    for key, value in element.items():
        if key not in dict_to_append:
            dict_to_append[key] = {}
        if type(value) is dict:
            dict_appendor(value, dict_to_append[key])
        else:
            if type(dict_to_append[key]) is dict:
                dict_to_append[key] = np.array(value)
            else:
                dict_to_append[key] = np.append(dict_to_append[key], value)


if __name__ == "__main__":
    dictionary = {
        "output_frequency_type": "time",
        "perturbation_factors": {
            "General_Sigma": 0.001,
            "Shapes": {
                "Rectangles": {
                    "Num": 10,
                    "Fiber_Content": [
                        -0.3,
                        0.3
                    ]
                },
                "Circles": {
                    "Num": 10,
                    "Fiber_Content": [
                        -0.3,
                        0
                    ]
                }
            }
        }
    }

    file = create_hdf5("test")
    write_dict_to_hdf5(file, dictionary)

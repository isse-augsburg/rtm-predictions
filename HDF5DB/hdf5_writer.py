import h5py
import numpy as np


def create_hdf5(filename):
    return h5py.File(filename, 'a')


# outfile - a h5py file or group, dict_to_save - a dictionary
def write_dict_to_hdf5(outfile, dict_to_save: dict):
    for name, value in dict_to_save.items():
        if type(value) is dict:
            try:
                grp = outfile.create_group(str(name))
            except ValueError:
                grp = outfile[str(name)]

            write_dict_to_hdf5(grp, value)
        elif type(value) is list and type(value[0]) is dict:
            try:
                outfile = outfile.create_group(str(name))
            except ValueError:
                outfile = outfile[str(name)]
            wrapped = wrap_list(value)
            write_dict_to_hdf5(outfile, wrapped)
        else:
            outfile.create_dataset(str(name), data=np.array(value))


def wrap_list(inputlist):
    new_dict = {}
    for element in inputlist:
        dict_appendor(element, new_dict)
    return new_dict


def dict_appendor(element, dict_to_append):
    for name, value in element.items():
        if name not in dict_to_append:
            dict_to_append[name] = {}
        if type(value) is dict:
            dict_appendor(value, dict_to_append[name])
        else:
            if type(dict_to_append[name]) is dict:
                dict_to_append[name] = np.array(value)
            else:
                dict_to_append[name] = np.append(dict_to_append[name], value)


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
        },
        "shapes": [
            {
                "Rectangle":
                    {
                        "fvc": -0.29499650602394134,
                        "height": 4.75,
                        "width": 6.875
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": -0.11555955416902425,
                        "height": 4.625,
                        "width": 5.25
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": 0.10998466945759067,
                        "height": 6.625,
                        "width": 6.125
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": 0.24736346628701705,
                        "height": 5.25,
                        "width": 2.375
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": -0.19367454313601146,
                        "height": 2.375,
                        "width": 6.875
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": -0.1039589830843318,
                        "height": 2.375,
                        "width": 5.25
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": -0.2460573018344987,
                        "height": 3.875,
                        "width": 1.125
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": 0.2916691508381806,
                        "height": 6.25,
                        "width": 6.0
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": -0.12815623764975065,
                        "height": 7.625,
                        "width": 2.625
                    }
            },
            {
                "Rectangle":
                    {
                        "fvc": 0.009141529067154486,
                        "height": 5.125,
                        "width": 7.5
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.23507349652462617,
                        "radius": 1.625
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.19819836709287625,
                        "radius": 1.375
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.21887138547611146,
                        "radius": 1.375
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.2381771208260994,
                        "radius": 1.75
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.0894160696915535,
                        "radius": 2.75
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.04709325030942252,
                        "radius": 2.375
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.2638248085877042,
                        "radius": 1.75
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.037285027564015194,
                        "radius": 1.5
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.24862836339506822,
                        "radius": 2.75
                    }
            },
            {
                "Circle":
                    {
                        "fvc": -0.2381629990714681,
                        "radius": 1.875
                    }
            }
        ]
    }

    file = create_hdf5("test10")
    write_dict_to_hdf5(file, dictionary)

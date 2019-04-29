import h5py
import numpy as np


def create_h5(filename):
    return h5py.File(filename+'.hdf5','a')


# f - a h5py file or group, d - a dictionary
def write_dict_to_Hdf5(f, d:dict):
   for k, v in d.items():
        if type(v) is dict:
            try:
                grp = f.create_group(str(k))
            except ValueError:
                grp = f[str(k)]
            
            write_dict_to_Hdf5(grp, v)
        elif type(v) is list and type(v[0]) is dict:
            try:
                f = f.create_group(str(k))
            except ValueError:
                f = f[str(k)]
            wrapped = wrap_list(v)
            write_dict_to_Hdf5(f,wrapped)
        else:
            if type(v) is str:
                f.create_dataset(str(k), data = v)
            f.create_dataset(str(k), data =  np.array(v))


def wrap_list(v):
    new_dict = {}
    for el in v:
        dict_appendor(el, new_dict)
    return new_dict

def dict_appendor(el, d):
    for k,v in el.items():
        if k not in d:
            d[k] = {}
        if type(v) is dict:
            dict_appendor(v, d[k])
        else:
            if type(d[k]) is dict:
                d[k] = np.array(v)
            else:
                d[k] = np.append(d[k], v)
            
           

if __name__ == "__main__":
    d = {
    "output_frequency_type": 0,
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
    "shapes": [{
            "Rectangle": {
                "fvc": -0.29499650602394134,
                "height": 4.75,
                "width": 6.875
            }
        },
        {
            "Rectangle": {
                "fvc": -0.11555955416902425,
                "height": 4.625,
                "width": 5.25
            }
        },
        {
            "Rectangle": {
                "fvc": 0.10998466945759067,
                "height": 6.625,
                "width": 6.125
            }
        },
        {
            "Rectangle": {
                "fvc": 0.24736346628701705,
                "height": 5.25,
                "width": 2.375
            }
        },
        {
            "Rectangle": {
                "fvc": -0.19367454313601146,
                "height": 2.375,
                "width": 6.875
            }
        },
        {
            "Rectangle": {
                "fvc": -0.1039589830843318,
                "height": 2.375,
                "width": 5.25
            }
        },
        {
            "Rectangle": {
                "fvc": -0.2460573018344987,
                "height": 3.875,
                "width": 1.125
            }
        },
        {
            "Rectangle": {
                "fvc": 0.2916691508381806,
                "height": 6.25,
                "width": 6.0
            }
        },
        {
            "Rectangle": {
                "fvc": -0.12815623764975065,
                "height": 7.625,
                "width": 2.625
            }
        },
        {
            "Rectangle": {
                "fvc": 0.009141529067154486,
                "height": 5.125,
                "width": 7.5
            }
        },
        {
            "Circle": {
                "fvc": -0.23507349652462617,
                "radius": 1.625
            }
        },
        {
            "Circle": {
                "fvc": -0.19819836709287625,
                "radius": 1.375
            }
        },
        {
            "Circle": {
                "fvc": -0.21887138547611146,
                "radius": 1.375
            }
        },
        {
            "Circle": {
                "fvc": -0.2381771208260994,
                "radius": 1.75
            }
        },
        {
            "Circle": {
                "fvc": -0.0894160696915535,
                "radius": 2.75
            }
        },
        {
            "Circle": {
                "fvc": -0.04709325030942252,
                "radius": 2.375
            }
        },
        {
            "Circle": {
                "fvc": -0.2638248085877042,
                "radius": 1.75
            }
        },
        {
            "Circle": {
                "fvc": -0.037285027564015194,
                "radius": 1.5
            }
        },
        {
            "Circle": {
                "fvc": -0.24862836339506822,
                "radius": 2.75
            }
        },
        {
            "Circle": {
                "fvc": -0.2381629990714681,
                "radius": 1.875
            }
        }
    ]
}

    f = create_h5("test8")
    write_dict_to_Hdf5(f, d)

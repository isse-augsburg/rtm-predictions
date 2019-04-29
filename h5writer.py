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
        elif type(v) is list:
            try:
                f = f.create_group(str(k))
            except ValueError:
                f = f[str(k)]
            wrapped = wrap_list(v)
            write_dict_to_Hdf5(f,wrapped)
        else:
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
    "shapes": [
        {
            "Rectangle": {
                "fvc": -0.23501463680736057,
                "height": 4.5,
                "width": 2.75
            }
        },
        {
            "Rectangle": {
                "fvc": -0.1361909449575696,
                "height": 3.375,
                "width": 1.5
            }
        },
        {
            "Rectangle": {
                "fvc": -0.10912324096799914,
                "height": 7.125,
                "width": 2.875
            }
        },
        {
            "Rectangle": {
                "fvc": -0.03555376111288022,
                "height": 7.875,
                "width": 3.375
            }
        },
        {
            "Rectangle": {
                "fvc": 0.1691010838914181,
                "height": 5.375,
                "width": 5.375
            }
        },
        {
            "Rectangle": {
                "fvc": 0.10456716763032725,
                "height": 2.25,
                "width": 6.5
            }
        },
        {
            "Rectangle": {
                "fvc": -0.017124812014985502,
                "height": 3.5,
                "width": 2.375
            }
        },
        {
            "Rectangle": {
                "fvc": 0.12466440618953961,
                "height": 6.75,
                "width": 6.0
            }
        },
        {
            "Rectangle": {
                "fvc": 0.17095740214883742,
                "height": 4.5,
                "width": 4.0
            }
        },
        {
            "Rectangle": {
                "fvc": 0.21031252562672303,
                "height": 1.375,
                "width": 2.125
            }
        },
        {
            "Circle": {
                "fvc": -0.23666634579863027,
                "radius": 2.75
            }
        },
        {
            "Circle": {
                "fvc": -0.2988952652804413,
                "radius": 1.875
            }
        },
        {
            "Circle": {
                "fvc": -0.1185185560688713,
                "radius": 2.0
            }
        },
        {
            "Circle": {
                "fvc": -0.10014007510585546,
                "radius": 1.375
            }
        },
        {
            "Circle": {
                "fvc": -0.19697543443220716,
                "radius": 2.25
            }
        },
        {
            "Circle": {
                "fvc": -0.2702415878159245,
                "radius": 2.625
            }
        },
        {
            "Circle": {
                "fvc": -0.058009654067942146,
                "radius": 2.5
            }
        },
        {
            "Circle": {
                "fvc": -0.1356738699648369,
                "radius": 1.125
            }
        },
        {
            "Circle": {
                "fvc": -0.1351815688151721,
                "radius": 2.125
            }
        },
        {
            "Circle": {
                "fvc": -0.21838550137776358,
                "radius": 1.375
            }
        }
    ]
}
    f = create_h5("test2")
    write_dict_to_Hdf5(f, d)

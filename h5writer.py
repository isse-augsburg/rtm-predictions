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
        else:
            f.create_dataset(str(k), data =  np.array(v))
           

if __name__ == "__main__":
    d = \
        {
        "General_Sigma": .001,
        "Shapes":
            {
                "Rectangles":
                    {
                        "Num": 1,
                        "Fiber_Content":
                            [-.3, .3]
                    },
                "Circles":
                    {
                        "Num": 0,
                        "Fiber_Content": [-.3, 0]
                    }
            }
        }
    k = \
        {
        "Shapes":
            {
                "Elipsies":
                    {
                        "Num": 1,
                        "Fiber_Content":
                            [-.3, .3]
                    },
            }
        }

    f = create_h5("test2")
    write_dict_to_Hdf5(f, d)
    write_dict_to_Hdf5(f, k)
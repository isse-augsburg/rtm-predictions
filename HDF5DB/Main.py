# import cProfile
import timeit

from HDF5DB.hdf5db_toolbox import HDF5DBToolbox


def test():
    h5db = HDF5DBToolbox()

    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p"
    )
    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-24_16-32-40_5000p"
    )
    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-29_10-45-18_5000p"
    )
    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-23_15-10-02_5000p"
    )
    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-24_11-51-48_5000p"
    )
    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-25_09-16-40_5000p"
    )
    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-26_16-59-08_6000p"
    )
    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-09-06_17-03-51_10000p"
    )
    h5db.force_save(
        "/cfs/share/cache/HDF5DB_Cache",
        "Unrestricted"
    )


if __name__ == "__main__":
    test()

# TODO
# Min/Max width/heigth of rectangles
# Same for circle: radius
# Total number of frames

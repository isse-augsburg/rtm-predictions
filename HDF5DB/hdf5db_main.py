from pathlib import Path

from HDF5DB.hdf5db_toolbox import HDF5DBToolbox


def create_index():
    h5db = HDF5DBToolbox()

    # h5db.add_objects_from_path(
    #     "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p"
    # )
    # h5db.add_objects_from_path(
    #     "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-24_16-32-40_5000p"
    # )
    # h5db.add_objects_from_path(
    #     "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-29_10-45-18_5000p"
    # )
    # h5db.add_objects_from_path(
    #     "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-23_15-10-02_5000p"
    # )
    # h5db.add_objects_from_path(
    #     "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-24_11-51-48_5000p"
    # )
    # h5db.add_objects_from_path(
    #     "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-25_09-16-40_5000p"
    # )
    # h5db.add_objects_from_path(
    #     "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-26_16-59-08_6000p"
    # )
    # h5db.add_objects_from_path(
    #     "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-09-06_17-03-51_10000p"
    # )
    h5db.add_objects_from_path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-11-08_15-40-44_5000p"
    )
    h5db.force_save(
        "/cfs/share/cache/HDF5DB_Cache",
        "Unrestricted"
    )


def load_something():
    hdf5db = HDF5DBToolbox()
    hdf5db.load(
        Path("Y:\cache\HDF5DB_Cache"),
        "Unrestricted"
    )
    hdf5db.show_objects()


if __name__ == "__main__":
    load_something()
    # create_index()

import cProfile
import timeit

from HDF5DB import HDF5DB


def test():
    test = HDF5DB()
    # test2 = HDF5Object("Data/0") #/2019-07-23_15-38-08_0
    # test.addObject(test2)
    # test.add_objects_from_path(
    #     "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p"
    # )
    # test.add_objects_from_path(
    #      "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-24_16-32-40_5000p"
    # )
    test.load(
        "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache/HDF5DB_Cache",
        "Unrestricted"
    )
    # test.add_objects_from_path(
    #     "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-29_10-45-18_5000p"
    # )
    # test.add_objects_from_path(
    #     "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-23_15-10-02_5000p"
    # )
    # test.add_objects_from_path(
    #     "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-24_11-51-48_5000p"
    # )
    # test.add_objects_from_path(
    #     "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-25_09-16-40_5000p"
    # )
    # test.add_objects_from_path(
    #     "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-08-26_16-59-08_6000p"
    # )
    # test.add_objects_from_path("Data")
    # test.showObjects()
    # test.save(
    #     "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache/HDF5DB_Cache",
    #     "Unrestricted"
    # )
    # test.load("Data", "HDF5DB")
    # test.show_selection_options()
    # test.showObjects()
    # test.select2("age", "2019-07-23_15-38-08", "=")
    test.select("fvc_circle", 0.2, ">")
    # test.select("output_frequency", 1, ">")
    # test.select("fibre_content_runners", -0.7, "=")
    # test.select("avg_level", 0.8, ">")
    # test.select("posy_circle", 50, "<")
    # test.show_objects()


# def main():
t = timeit.Timer("test()", "from __main__ import test")
print(t.timeit(1))
# cProfile.run('test()')

# if __name__ == "__main__":
#     test()
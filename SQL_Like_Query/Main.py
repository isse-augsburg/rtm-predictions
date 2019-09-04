import cProfile
import timeit

from HDF5DB import HDF5DB


def test():
    test = HDF5DB()
    # test2 = HDF5Object("Data/0") #/2019-07-23_15-38-08_0
    # test.addObject(test2)
    # test.add_objects_from_path("/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p")
    # test.add_objects_from_path("Data")
    # test.showObjects()
    # test.save("Data", "HDF5DB")
    test.load("Data", "HDF5DB")
    # test.show_selection_options()
    # test.showObjects()
    # test.select2("age", "2019-07-23_15-38-08", "=")
    test.select2("fvc_circle", 0.2, ">")
    # test.select("output_frequency", 1, ">")
    #test.select("fibre_content_runners", -0.7, "=")
    #test.select("avg_level", 0.8, ">")
    #test.select("posy_circle", 50, "<")
    # test.show_objects()

# def main():
# t = timeit.Timer("test()", "from __main__ import test")
# print(t.timeit(1000))
# cProfile.run('test()')

if __name__ == "__main__":
    test()

# TODO
#  Funkion mit Rückgabewert Liste von Pfaden
#  Cachedatei mit alten Suchen erstellen, nutzen
#  Variablen Funktion
#  Operator
#  Prüfen bzw. casten von evtl. String zu Integer

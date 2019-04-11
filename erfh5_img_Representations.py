import h5py
import numpy as np
from os import listdir, walk
import os
from PIL import Image
from multiprocessing import Pool


def get_paths_to_files(root_directory):
    dataset_filenames = []
    for (dirpath, dirnames, filenames) in walk(root_directory):
        if filenames:
            filenames = [dirpath + '/' + f for f in filenames]
            dataset_filenames.extend(filenames)
    return dataset_filenames


def create_images_for_file(filename, main_folder="/cfs/home/s/c/schroeni/Git/tu-kaiserslautern-data/Images/"):
    f = h5py.File(filename, 'r')

    
    folder_name = filename.split('/')[-1].replace(".", "")
    print(folder_name)
    print(filename)
    os.mkdir(main_folder+folder_name)
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _coords = coord_as_np_array[:, :-1]
    _coords = normalize_coords(_coords)
    all_states = f['post']['singlestate']
    filling_factors_at_certain_times = list()
    for i,state in enumerate(all_states):
        try:
            filling = f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
            time = f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()][0]
            filling = filling.flatten()

            non_zeros = np.count_nonzero(filling)
            state_count = np.shape(filling)[0]
            filling_percentage = non_zeros/state_count
            
            f_name = "{:05d}".format(i) +"_"+'{:1.6f}'.format(filling_percentage)+"_" +'{:010.6f}'.format(time)
            #print(f_name)
            create_img(norm_coords=_coords, data=filling, folder=main_folder+folder_name, filename=f_name)
        except KeyError as e:
            print(e)
            continue
    #states_as_list = [x[-5:] for x in list(all_states.keys())]
    #flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    
      

    return 


def normalize_coords(coords):
    max_c = np.max(coords)
    min_c = np.min(coords)
    coords = coords - min_c
    coords = coords /(max_c-min_c)
    return coords

def create_img(target_shape = (264,264), norm_coords=None, data=None,folder ="", filename=""):
    if norm_coords is None or data is None or folder is None or filename is None:
        print("ERROR")
        return
    assert np.shape(norm_coords)[0] == np.shape(data)[0]
    
    img = Image.new(mode='P', size=target_shape)
    pixels = img.load()
    for i, value in enumerate(data):
        coord = norm_coords[i]
        x,y = int(round(coord[0]*(target_shape[0]-1))), int(round(coord[1]*(target_shape[1]-1)))
        #print(x,y)
        pixels[x,y] = (int(value*255))
    #img.show("test")
    img.save(str(folder)+"/"+str(filename)+".png")
if __name__ == "__main__":
    folder_path = "/cfs/share/data/RTM/Lautern/clean_erfh5"
    #main_folder="Images/"
    data_list = get_paths_to_files(folder_path)
    p = Pool(5)
    p.map(create_images_for_file,data_list)
"""  for file in data_list:
        create_images_for_file(file) """
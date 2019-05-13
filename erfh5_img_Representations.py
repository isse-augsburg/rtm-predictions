import h5py
import numpy as np
from os import listdir, walk
import os
from PIL import Image, ImageColor
from multiprocessing import Pool
from tqdm import tqdm




def get_paths_to_files(root_directory):
    dataset_filenames = []
    for (dirpath, dirnames, filenames) in walk(root_directory):
        if filenames:
            filenames = [dirpath + '/' + f for f in filenames]
            dataset_filenames.extend(filenames)
    return dataset_filenames


def create_images_for_file(filename, main_folder="/run/user/1002/gvfs/smb-share:server=137.250.170.56,share=home/s/c/schroeni/Data/Images_Debug/"):
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
    print(">>> Num States: ", len(all_states))
    for i,state in tqdm(enumerate(all_states), total=np.shape(all_states)[0]):
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
            #create_np_image(norm_coords=_coords, data=filling, folder=main_folder+folder_name, filename=f_name)
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

def create_img(target_shape = (150,150), norm_coords=None, data=None,folder ="", filename=""):
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
        #color = 'hsl(%d, 100%%,%d%%)' %(value*120, value*50)
        #color = ImageColor.getrgb(color)
        pixels[x,y] = int(value*255)
    #img.show("test")
    img.save(str(folder)+"/"+str(filename)+".png")

# WORK IN PROGRESS
def create_np_image(target_shape = (150,150), norm_coords=None, data=None,folder ="", ):
    if norm_coords is None or data is None or folder is None:
        print("ERROR")
        return
    assert np.shape(norm_coords)[0] == np.shape(data)[0]
    
    arr = np.zeros(target_shape)


    data = np.expand_dims(data, axis=1)
    coords_value = np.append(norm_coords,data ,axis=1)
    coords_value[:,0] = coords_value[:,0]*(target_shape[0]-1)
    coords_value[:,1] = coords_value[:,1]*(target_shape[1]-1)
    coords_value[:,2] = coords_value[:,2]*255
    coords_value = coords_value.astype(np.int)
    arr[coords_value[:,0],coords_value[:,1]] = coords_value[:,2] 
    
    
    return arr



if __name__ == "__main__":

    create_images_for_file("/run/user/1002/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-04-23_10-23-20_200p/91/2019-04-23_10-23-20_91_RESULT.erfh5")
    """ folder_path = "/cfs/share/data/RTM/Lautern/clean_erfh5"
    #main_folder="Images/"
    data_list = get_paths_to_files(folder_path)
    p = Pool(5)
    p.map(create_images_for_file,data_list)
    for file in data_list:
        create_images_for_file(file) """ 
import h5py 
import numpy as np 
from PIL import Image
from os import listdir, walk
from os.path import isdir
import random

#returns a sequence of simulation steps as data and the filling percentage of the last step as label 
def get_index_sequence(filename): 
    indices = [10, 20, 30, 40, 50]

    f = h5py.File(filename, 'r')
    # Cut off last column (z), since it is filled with 1s anyway
    all_states = f['post']['singlestate']
    filling_factors_at_certain_times = list()
    
    for state in all_states:
        try:
            filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
            
        except KeyError:
            continue
        
        filling_factors_at_certain_times.append(filling_factor)   

    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    last_filling = flat_fillings[-1]
    non_zeros = np.count_nonzero(last_filling)
    state_count = np.shape(last_filling)[0]
    filling_percentage = np.array(non_zeros/state_count)
        
    try:
        flat_fillings = [flat_fillings[j] for j in indices]
    except IndexError:
        return None 
    
    return [(flat_fillings, filling_percentage)]

def get_all_sequences_for_file(filename):
    all_sequences = list()
    t_begin = 0
    t_end = 10
    t_delta = 5 
    t_target_offset = 1
    t_sequence_distance = 10
    t_final = 1000
    
    while t_end + t_target_offset < t_final:
        try:
            instance = __get_fillings_at_times(filename, t_begin,t_end,t_delta,t_end+t_target_offset)
        except Exception:
            break
        
        t_begin = t_begin +t_sequence_distance
        t_end = t_end + t_sequence_distance

        if instance is None: 
            continue 
        else: 
            all_sequences.append(instance)

    
    if len(all_sequences) == 0:
        return None 

    return all_sequences


def __get_fillings_at_times(filename, t_start, t_finish, t_delta, t_target):
    t_now = t_start

    try:
        f = h5py.File(filename, 'r')
    except OSError:
        print(">>> ERROR: FILE", filename, "COULD NOT BE OPEND BY H5PY. THIS IS BAD. BIG OOooOF")
        return None 

    all_states = f['post']['singlestate']
    filling_factors_at_certain_times = list()
    filling_percentage = -1

    final_state = ""
    for s in all_states:
        final_state = s 

    end_time = f['post']['singlestate'][final_state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]
    if(end_time < t_target):
        raise Exception

    for state in all_states:
        try:
            time = f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]
            
            if time >= t_target:
                target_fillingstate = filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
                non_zeros = np.count_nonzero(target_fillingstate)
                state_count = np.shape(target_fillingstate)[0]
                filling_percentage = np.array(non_zeros/state_count)
                t_target = 9999999
                break
            if(time >= t_finish):
                continue
            if(time >= t_now):
                filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
                filling_factors_at_certain_times.append(filling_factor)   
                t_now += t_delta
            
        except KeyError:
            continue


    if(t_target != 9999999 or filling_factors_at_certain_times.__len__() != (t_finish -t_start)/t_delta):
        #print("Didn't",len(filling_factors_at_certain_times), t_target, filling_percentage)
        return None 


    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]

    #print("Worked",len(filling_factors_at_certain_times), t_target, filling_percentage)
    return (flat_fillings, filling_percentage)

def get_single_states_and_fillings(filename):
    instances = []
    f = h5py.File(filename, 'r')
    all_states = f['post']['singlestate']
    
    filling_factors_at_certain_times = list()
    
    for state in all_states:
        try:
            filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]

        except KeyError:
            continue
        
        filling_factors_at_certain_times.append(filling_factor)   

    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    single_states = [(data, data) for data in flat_fillings]

    
    return single_states

def get_sensordata_and_filling_percentage(file, until=250, frm = 50):
    f = h5py.File(file, 'r')
    try:
        pressure_array = f['post']['multistate']['TIMESERIES1']['multientityresults']['SENSOR']['PRESSURE']['ZONE1_set1']['erfblock']['res'][()]
        all_states = f['post']['singlestate']
        _tmp = [state for state in all_states]
        last_filling =  f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
        non_zeros = np.count_nonzero(last_filling)
        state_count = np.shape(last_filling)[0]
        filling = np.floor(non_zeros/state_count)
        filling_percentage = np.array((filling, 1 - filling), dtype=np.long)

    except KeyError:
        return None

    if(np.shape(pressure_array)[0] < until):
        return None
    pressure_array = pressure_array[frm:until,:,:]
    pressure_array = np.squeeze(pressure_array)

    #print(np.shape(pressure_array), filling_percentage)

    return([(pressure_array,filling_percentage)])




    

def get_image_state_sequence(folder, start_state=0, end_state=100, step=5, label_offset=3):
    filelist = get_filelist_within_folder(folder)
    if len(filelist) == 0:
        return None
    filelist.sort()
    state_list = list()

    f = 0

   
    c_state = start_state

    label  =None

    while(f < len(filelist)):
        f_meta = filelist[f]
        f_split = f_meta.split("/")[-1].split("_")
        state_index = int(f_split[0])


        if state_index <= c_state:
            state_list.append(load_image(f_meta))
        f+=1
        c_state+=step
        
        if(c_state > end_state):
            f+=label_offset-1
            if(f < len(filelist)):
                f_meta = filelist[f]
                label = load_image(f_meta)

            break

    if label is None:
        return None
    
    data = np.stack(state_list)
    if np.shape(data)[0] !=int((end_state - start_state)/step) +1:
        return None

    return [(data, label)]


def get_sensordata_and_filling_percentage_v2(file, until=400, frm = 0):
    f = h5py.File(file, 'r')
    try:
        pressure_array = f['post']['multistate']['TIMESERIES1']['multientityresults']['SENSOR']['PRESSURE']['ZONE1_set1']['erfblock']['res'][()]
        all_states = f['post']['singlestate']
        _tmp = [state for state in all_states]
        last_filling =  f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
        non_zeros = np.count_nonzero(last_filling)
        state_count = np.shape(last_filling)[0]
        filling_percentage = np.array(non_zeros/state_count)  

    except KeyError:
        return None

    if(np.shape(pressure_array)[0] < until):
        return None
    pressure_array = pressure_array[frm:until,:,:]
    pressure_array = np.squeeze(pressure_array)

    #print(np.shape(pressure_array), filling_percentage)

    return([(pressure_array,filling_percentage)])


def get_sensordata_and_flowfront(file):
    f = h5py.File(file, 'r')
    instances = []
    try:
        pressure_array = f['post']['multistate']['TIMESERIES1']['multientityresults']['SENSOR']['PRESSURE']['ZONE1_set1']['erfblock']['res'][()]
        all_states = f['post']['singlestate']

        filling_factors_at_certain_times = [f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()] for state in all_states]
        flat_fillings = np.squeeze(filling_factors_at_certain_times)
    except KeyError:
        return None
    for state, filling in zip(all_states, flat_fillings):
            try:
                s = state.replace("state", '')
                state_num = int(s)
                sensordata = np.squeeze(pressure_array[state_num-1])
                #print(state_num, np.shape(filling), np.shape(sensordata))
                instances.append((sensordata, filling))
            except IndexError:
                continue
    if(len(instances) == 0):
        return None
    return instances

def get_image_percentage(folder):
    filelist = get_filelist_within_folder(folder)
    if len(filelist) == 0:
        return None
    random.shuffle(filelist)
    
    ret_list = []
    for el in filelist:
        f_split = el.split("/")[-1].split("_")
        percentage = float(f_split[1])
        dat = load_image(el)
        ret_list.append((dat,np.array(percentage)))
    
    return ret_list





def load_image( f_name ) :
    img = Image.open( f_name )
    img.load()
    data = np.asarray( img, dtype="float32" )
    return data

# Functions to gather data within a folder
def get_filelist_within_folder(root_directory):
    dataset_filenames = []
    for dirs in root_directory:
        
        for (dirpath, _, filenames) in walk(dirs):
            if filenames: 
                filenames = [dirpath + '/' + f for f in filenames if f.endswith('.erfh5')]
                dataset_filenames.extend(filenames)
    
    return dataset_filenames  



def get_folders_within_folder(root_directory):
    folders = list()

    for dirs in root_directory:

        for (dirpath, dirnames, _) in walk(root_directory):
            l = [dirpath+ '/' + f for f in dirnames]
            folders.extend(l)
            break
    return folders


if __name__ == "__main__":

    files = get_filelist_within_folder('/run/user/1002/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/')
    for f in files:
        r = get_sensordata_and_flowfront(f)
        if r is not None:
            for d in r:
                print(np.shape(d[0]), np.shape(d[1]))
        
    
    
    """ folders = get_folders_within_folder('/cfs/home/s/c/schroeni/Git/tu-kaiserslautern-data/Images/')
    for folder in folders:
        instances = get_image_state_sequence(folder)
        if instances is None:
            
            print("None")
        else:
            print(np.shape(instances[0][0]))

 """
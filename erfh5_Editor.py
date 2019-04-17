import h5py
import numpy as np



def add_filling_percentage(h5pyfile, grp):
    f = h5pyfile
    all_states = f['post']['singlestate']
    _tmp = [state for state in all_states]
    last_filling =  f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
    non_zeros = np.count_nonzero(last_filling)
    state_count = np.shape(last_filling)[0]
    filling_percentage = np.array(non_zeros/state_count)  
    grp.create_dataset('filling_percentage', data=filling_percentage)
    

def add_finished_time(h5pyfile, grp):
    f = h5pyfile
    all_states = f['post']['singlestate']
    _tmp = [state for state in all_states]
    time =  f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]
    grp.create_dataset('finish_time', data=time)


if __name__ == "__main__":
    f= '/home/niklas/Documents/Data/2019-04-16_08-36-09_2_RESULT.erfh5'

    f = h5py.File(f, 'r+')
    try:
        grp = f.create_group("meta_data")
    except ValueError:
        grp = f['meta_data']

    add_finished_time(f,grp)
    
    #f.create_dataset('nice_value-v1', data=np.array(666))

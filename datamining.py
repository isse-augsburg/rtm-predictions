import h5py 
import matplotlib.pyplot as plt 
import matplotlib
import os 
import time 
import numpy as np 
import torch
from multiprocessing import Process, Queue
from multiprocessing import cpu_count
from functools import partial
from os import listdir, walk 
import queue 
import threading 
import random 
from enum import Enum


class NoSequenceException(Exception):
    pass


#returns a list of all file paths in a root directory including sub-directories
def get_paths_to_files(root_directory):
    dataset_filenames = []
    for (dirpath, dirnames, filenames) in walk(root_directory):
        if filenames: 
            filenames = [dirpath + '/' + f for f in filenames]
            dataset_filenames.extend(filenames)
    return dataset_filenames   

def get_states_and_fillings(filename):
        
    f = h5py.File(filename, 'r')

    
    all_states = f['post']['singlestate']
    j = ""
    filling_factors_at_certain_times = list()
    """ for k in all_states:
        j = k """
    filling_percentages = list()
    labels = list()
    
    for state in all_states:
        try:
            last_filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
            label = f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]
            last_filling_factor = last_filling_factor.flatten()
            non_zeros = np.count_nonzero(last_filling_factor)
            state_count = np.shape(last_filling_factor)[0]
            filling_percentage = np.array(non_zeros/state_count)
            filling_percentages.append(filling_percentage)
            labels.append(label)

        except KeyError as e:
            #print("KeyError in file", filename)
            continue

    plt.plot(labels, filling_percentages)
    plt.show()
    



def get_all_sequences_for_file(filename, t_begin, t_end, t_delta, t_target_offset, t_sequence_distance, t_final):
    all_sequences = list()

    
    try:
        while t_end + t_target_offset < t_final:
            instance = get_fillings_at_times(filename, t_begin,t_end,t_delta,t_end+t_target_offset)
            t_begin = t_begin +t_sequence_distance
            t_end = t_end + t_sequence_distance
            all_sequences.append(instance)
    except NoSequenceException:
        pass
    
    if len(all_sequences) == 0:
        raise NoSequenceException

    return all_sequences


def get_fillings_at_times(filename, t_start, t_finish, t_delta, t_target):
    t_now = t_start
    f = h5py.File(filename, 'r')
    all_states = f['post']['singlestate']
    filling_factors_at_certain_times = list()
    filling_percentage = -1
 
    
    for state in all_states:
        try:
            time = f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]
            
            if time >= t_target:
                target_fillingstate = filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
                non_zeros = np.count_nonzero(target_fillingstate)
                state_count = np.shape(target_fillingstate)[0]
                filling_percentage = np.array(non_zeros/state_count)
                t_target = 9999999
            if(time >= t_finish):
                continue
            if(time >= t_now):
                filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
                filling_factors_at_certain_times.append(filling_factor)   
                t_now += t_delta
            
        except KeyError as e:
            continue
        
    
    #label = f['post']['singlestate'][j]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]


    if(t_target != 9999999 or filling_factors_at_certain_times.__len__() != (t_finish -t_start)/t_delta):
        #print("Didn't",len(filling_factors_at_certain_times), t_target, filling_percentage)
       raise NoSequenceException


    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
   
    #print("Worked",len(filling_factors_at_certain_times), t_target, filling_percentage)
    return flat_fillings, filling_percentage    
        


paths = get_paths_to_files('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/')

for file in paths:
    get_states_and_fillings(file)
    #get_fillings_at_times(file,50,90,2,100)
    """ try:
        print(len(get_all_sequences_for_file(file,0,50,2,3,10,1000)))
    except NoSequenceException as E:
        print("big oof.") """
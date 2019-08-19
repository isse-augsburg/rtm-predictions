from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

from Pipeline import erfh5_pipeline as pipeline, data_loaders as dl, data_loader_sensor as dls, data_loaders_IMG as dli, \
    data_gather as dg

from Trainer.evaluation import plot_predictions_and_label
from Trainer.Generic_Trainer import Master_Trainer
from Trainer.evaluation import Binary_Classification_Evaluator

import torch
import traceback
from torch import nn
from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from Models.custom_loss import FocalLoss

from Models.flow_front_to_fiber_fraction_model import FlowfrontToFiberfractionModel
from Models.erfh5_DeconvModel import DeconvModel
import os
import numpy as np
from PIL import Image
import time
import threading

#TODO
batchsize = 1
max_Q_len = 64
epochs = 100




### DEBUG
data_root = Path('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes')
cache_path = "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache"
###



# paths = [data_root / '2019-04-23_13-00-58_200p']#, data_root / '2019-04-23_10-23-20_200p']
path = data_root / '2019-07-23_15-38-08_5000p'
# path = data_root / '2019-05-17_16-45-57_3000p' / '0'
path = data_root / '2019-07-23_15-38-08_5000p'
# path = data_root / '2019-05-17_16-45-57_3000p'


#Debug path
path = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p'
paths = [path]




def create_dataGenerator_index_sequence():
    try:
        generator = pipeline.ERFH5_DataGenerator(
            '/cfs/share/data/RTM/Lautern/clean_erfh5/', data_processing_function=dl.get_index_sequence,
            data_gather_function=dg.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs, max_queue_length=2048, num_validation_samples=10)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()

    return generator


def create_dataGenerator_single_state():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_paths='/cfs/share/data/RTM/Lautern/clean_erfh5/',
                                                 data_processing_function=dl.get_single_states_and_fillings,
                                                 data_gather_function=dg.get_filelist_within_folder,
                                                 batch_size=batchsize, epochs=epochs, max_queue_length=2048,
                                                 num_validation_samples=3)
    except Exception as e:
        print("Fatal Error:", e)
        exit()
    return generator


def create_dataGenerator_pressure_sequence():
    try:
        
        generator = pipeline.ERFH5_DataGenerator(
            paths, data_processing_function=dls.get_sensordata_and_filling_percentage,
            data_gather_function=dg.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len, num_validation_samples=70)

    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()

    return generator



def get_comment():
    return "New sensorgrid. Experimenting with different settings."


if __name__ == "__main__":
    
    print(">>> INFO: Generating Model")
    model = ERFH5_PressureSequence_Model()
    print(">>> INFO: Generating Generator")
    generator = create_dataGenerator_pressure_sequence()
    print(">>> INFO: Model to GPU")
    model = nn.DataParallel(model).to('cuda:0')
    print(">>> INFO: Generating Trainer")
    #train_wrapper = Master_Trainer(model, generator, loss_criterion=torch.nn.BCELoss(), comment=get_comment(),
                                  #learning_rate=0.0001, classification_evaluator=Binary_Classification_Evaluator())
    train_wrapper = Master_Trainer(model, generator, loss_criterion=FocalLoss(gamma=0), comment=get_comment(),
                                  learning_rate=0.0001, classification_evaluator=Binary_Classification_Evaluator())
    print(">>> INFO: The Training Will Start Shortly")

    train_wrapper.start_training()
    #train_wrapper.save_model('/cfs/home/l/o/lodesluk/models/crnn_1505_1045.pt')
    #print("Model saved.")
   

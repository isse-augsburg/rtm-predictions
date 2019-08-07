from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

from Pipeline import erfh5_pipeline as pipeline, data_loaders as dl, data_loader_sensor as dls, data_loaders_IMG as dli, \
    data_gather as dg

from Trainer.Evaluation import plot_predictions_and_label
from Trainer.Generic_Trainer import Master_Trainer
from Trainer.evaluation import Binary_Classification_Evaluator

import torch
import traceback
from torch import nn
from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from Models.flow_front_to_fiber_fraction_model import FlowfrontToFiberfractionModel
import os
import numpy as np
from PIL import Image
import time
import threading

# savePath does not belong here, since its user dependent.
if os.name == 'nt':
    data_root = Path(r'Y:\data\RTM\Lautern\output\with_shapes')
    savepath = Path(r'C:\Users\stiebesi\code\saved_models')
else:
    data_root = Path('/cfs/share/data/RTM/Lautern/output/with_shapes')
    savepath = Path('/cfs/home/s/t/stiebesi/code/saved_models')


### DEBUG
data_root = Path('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes')
cache_path = "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache"
###



# paths = [data_root / '2019-04-23_13-00-58_200p']#, data_root / '2019-04-23_10-23-20_200p']
path = data_root / '2019-04-23_13-00-58_200p'
# path = data_root / '2019-05-17_16-45-57_3000p' / '0'
# path = data_root / '2019-06-05_15-30-52_1050p'
# path = data_root / '2019-05-17_16-45-57_3000p'
paths = [path]
# =======
# from Models.custom_loss import focal_loss, FocalLoss


# batchsize = 1
# max_Q_len = 512
# epochs = 70
# #path = ['/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/']
# path = ['/cfs/share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/', '/cfs/share/data/RTM/Lautern/output/with_shapes/2019-04-23_10-23-20_200p']
# #path = ['/cfs/share/data/RTM/Lautern/output/with_shapes']
# >>>>>>> model_trainer.py


def create_dataGenerator_pressure_flowfront():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_paths=paths, num_validation_samples=100,
                                                 batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len,
                                                 data_processing_function=dli.get_sensordata_and_flowfront,
                                                 data_gather_function=dg.get_filelist_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    return generator


def create_dataGenerator_Pressure_percentage():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_paths=paths, num_validation_samples=10,
                                                 batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len,
                                                 data_processing_function=dls.get_sensordata_and_filling_percentage,
                                                 data_gather_function=dg.get_filelist_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    return generator


def create_dataGenerator_IMG_percentage():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_paths="/cfs/home/s/c/schroeni/Data/Images",
                                                 num_validation_samples=100,
                                                 batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len,
                                                 data_processing_function=dli.get_image_percentage,
                                                 data_gather_function=dg.get_folders_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    return generator


def create_dataGenerator_IMG():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_paths="/cfs/home/s/c/schroeni/Git/tu-kaiserslautern-data/Images",
                                                 batch_size=batchsize, epochs=epochs, max_queue_length=2048,
                                                 data_processing_function=dli.get_image_state_sequence,
                                                 data_gather_function=dg.get_folders_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()
    return generator


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
            batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len, num_validation_samples=4)
        """ generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function = dl.get_all_sensor_sequences, data_gather_function = dl.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs ,max_queue_length=4096, num_validation_samples=250) """
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()

    return generator


def get_comment():
    return "Sensor values are now correctly scaled"


def create_datagenerator_flow_front_to_permeabilities(batch_size=1, num_validation_samples=10,
                                                      num_workers=20, max_Q_len=512, epochs=1000):
    try:
        generator = pipeline.ERFH5_DataGenerator(data_paths=paths,
                                                 data_processing_function=dli.get_images_of_flow_front_and_permeability_map,
                                                 data_gather_function=dg.get_filelist_within_folder,
                                                 batch_size=batch_size, epochs=epochs, max_queue_length=max_Q_len,
                                                 num_validation_samples=num_validation_samples, num_workers=num_workers, cache_path=cache_path)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()

    return generator


if __name__ == "__main__":
    print(">>> INFO: Generating Generator")
    generator = create_datagenerator_flow_front_to_permeabilities(batch_size=2,
                                                                  num_validation_samples=1,
                                                                  num_workers=6,
                                                                  max_Q_len=2048,
                                                                  epochs=1000)
    print("Generator finished")
    for i, (inputs, label) in enumerate(generator):
        print(i, np.shape(inputs), np.shape(label), len(generator.batch_queue), len(generator.path_queue), threading.active_count())
       
       

 
    """    print(">>> INFO: Generating Model")
    model = FlowfrontToFiberfractionModel()
    print(">>> INFO: Model to GPU")
    model = nn.DataParallel(model).to('cuda:0')

    train_wrapper = Master_Trainer(model, generator,
                                   comment=get_comment(),
                                   loss_criterion=torch.nn.MSELoss(),
                                   # loss_criterion=pixel_wise_loss_multi_input_single_label,
                                   savepath=savepath / 'flow_front_perm.pt',
                                   learning_rate=0.0001,
                                   calc_metrics=False,
                                   train_print_frequency=2,
                                   eval_frequency=100,
                                   eval_func=plot_predictions_and_label)
    print(">>> INFO: The Training Will Start Shortly")

    train_wrapper.start_training()
    train_wrapper.save_model('/cfs/home/l/o/lodesluk/models/crnn_1505_1045.pt')
    print("Model saved.")
    """
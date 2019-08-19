from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

from Pipeline import erfh5_pipeline as pipeline, data_loaders as dl, data_loader_sensor as dls, data_loaders_IMG as dli, \
    data_gather as dg


from Trainer.Generic_Trainer import Master_Trainer
from Trainer.evaluation import Sensor_Flowfront_Evaluator

import torch
import traceback
from torch import nn
from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from Models.flow_front_to_fiber_fraction_model import FlowfrontToFiberfractionModel
from Models.erfh5_DeconvModel import DeconvModel
import os
import numpy as np
from PIL import Image
import time
import threading


data_root = Path('/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes')
cache_path = "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache"
path = data_root / '2019-07-23_15-38-08_5000p'

paths = [path]


def create_dataGenerator_pressure_flowfront():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_paths=paths, num_validation_samples=1000,
                                                 batch_size=32, epochs=50, max_queue_length=8096,
                                                 data_processing_function=dli.get_sensordata_and_flowfront,
                                                 data_gather_function=dg.get_filelist_within_folder, num_workers=4, cache_path=cache_path)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    return generator





def get_comment():
    return "Hallo"


if __name__ == "__main__":
    print(">>> INFO: Generating Generator")
    generator = create_dataGenerator_pressure_flowfront() 
    print(">>> INFO: Generating Model")
    model = DeconvModel()
    print(">>> INFO:  Model to GPU")
    model = nn.DataParallel(model).to('cuda:0')

    train_wrapper = Master_Trainer(model, generator,
                                   comment=get_comment(),
                                   loss_criterion=torch.nn.MSELoss(),
                                   # loss_criterion=pixel_wise_loss_multi_input_single_label,
                                   savepath=None,
                                   learning_rate=0.0001,
                                   calc_metrics=False,
                                   train_print_frequency=10,
                                   eval_frequency=100,
                                   classification_evaluator = Sensor_Flowfront_Evaluator())
    print(">>> INFO: The Training Will Start Shortly")

    train_wrapper.start_training()
    print("Model saved.")
 
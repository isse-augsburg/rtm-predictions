import erfh5_pipeline as pipeline
from Generic_Trainer import Master_Trainer
import data_loaders as dl
import torch
import traceback
from torch import nn
from erfh5_RNN_models import ERFH5_RNN
from erfh5_autoencoder import erfh5_Distributed_Autoencoder
from erfh5_ConvModel import erfh5_Conv3d
from erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from erfh5_DeconvModel import DeconvModel

batchsize = 256
max_Q_len= batchsize * 16
epochs = 80
#path = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/'
path = ['/cfs/share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/', '/cfs/share/data/RTM/Lautern/output/with_shapes/2019-04-23_10-23-20_200p'] 


def create_dataGenerator_pressure_flowfront():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path= path,num_validation_samples=100,
                                    batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len, data_processing_function=dl.get_sensordata_and_flowfront, data_gather_function=dl.get_filelist_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    return generator

def create_dataGenerator_Pressure_percentage():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path= path,num_validation_samples=10,
                                    batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len, data_processing_function=dl.get_sensordata_and_filling_percentage, data_gather_function=dl.get_filelist_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    return generator

def create_dataGenerator_IMG_percentage():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path= "/cfs/home/s/c/schroeni/Data/Images",num_validation_samples=100,
                                    batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len, data_processing_function=dl.get_image_percentage, data_gather_function=dl.get_folders_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    return generator

def create_dataGenerator_IMG():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path= "/cfs/home/s/c/schroeni/Data/Images",
                                    batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len, data_processing_function=dl.get_image_state_sequence, data_gather_function=dl.get_folders_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    return generator


def create_dataGenerator_index_sequence():
    try:
        generator = pipeline.ERFH5_DataGenerator(
        '/cfs/share/data/RTM/Lautern/clean_erfh5/', data_processing_function = dl.get_index_sequence, data_gather_function = dl.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs ,max_queue_length=max_Q_len, num_validation_samples=10)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        traceback.print_exc()
        exit()
    
    return generator

def create_dataGenerator_single_state():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path='/cfs/share/data/RTM/Lautern/clean_erfh5/', data_processing_function=dl.get_single_states_and_fillings,
        data_gather_function=dl.get_filelist_within_folder, batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len, num_validation_samples=3)
    except Exception as e:
        print("Fatal Error:", e)
        traceback.print_exc()
        exit() 
    return generator

def create_dataGenerator_pressure_sequence(): 
    try: 
        generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function = dl.get_sensordata_and_filling_percentage, data_gather_function = dl.get_filelist_within_folder,
            batch_size=512, epochs=800 ,max_queue_length=2048, num_validation_samples=20)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()
    
    return generator

if __name__ == "__main__":
    #torch.cuda.set_device(0)
    #generator = create_dataGenerator_IMG()
    #generator = create_dataGenerator_index_sequence()
    #generator = create_dataGenerator_single_state()
    #model = ERFH5_PressureSequence_Model()
    #generator = create_dataGenerator_pressure_sequence()


    print(">>> INFO: Generating Model")
    model = DeconvModel()
    print(">>> INFO: Generating Generator")
    generator = create_dataGenerator_pressure_flowfront()
    print(">>> INFO: Model to GPU")
    model = nn.DataParallel(model).to('cuda:0')
    print(">>> INFO: Generating Trainer")
    train_wrapper = Master_Trainer(model, generator, loss_criterion=torch.nn.MSELoss(), eval_func=dl.save_numpy_as_image)
    print(">>> INFO: The Training Will Start Shortly")

    
   

    

   

    train_wrapper.start_training()
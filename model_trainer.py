import erfh5_pipeline as pipeline
from Generic_Trainer import Master_Trainer
import data_loaders as dl
import torch
from torch import nn
from erfh5_RNN_models import ERFH5_RNN
from erfh5_autoencoder import erfh5_Distributed_Autoencoder
from erfh5_ConvModel import erfh5_Conv3d
from erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model

batchsize = 1
epochs = 100
savepath = '/cfs/home/l/o/lodesluk/models/crnn.pt'
#path = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/'
path = ['/cfs/share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/', '/cfs/share/data/RTM/Lautern/output/with_shapes/2019-04-23_10-23-20_200p'] 

def create_dataGenerator_IMG():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path= "/cfs/home/s/c/schroeni/Git/tu-kaiserslautern-data/Images",
                                    batch_size=batchsize, epochs=epochs, max_queue_length=2048, data_processing_function=dl.get_image_state_sequence, data_gather_function=dl.get_folders_within_folder)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()
    return generator


def create_dataGenerator_index_sequence():
    try:
        generator = pipeline.ERFH5_DataGenerator(
        '/cfs/share/data/RTM/Lautern/clean_erfh5/', data_processing_function = dl.get_index_sequence, data_gather_function = dl.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs ,max_queue_length=2048, num_validation_samples=10)
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()
    
    return generator

def create_dataGenerator_single_state():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path='/cfs/share/data/RTM/Lautern/clean_erfh5/', data_processing_function=dl.get_single_states_and_fillings,
        data_gather_function=dl.get_filelist_within_folder, batch_size=batchsize, epochs=epochs, max_queue_length=2048, num_validation_samples=3)
    except Exception as e:
        print("Fatal Error:", e)
        exit() 
    return generator

def create_dataGenerator_pressure_sequence(): 
    try: 
        generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function = dl.get_sensordata_and_filling_percentage, data_gather_function = dl.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs ,max_queue_length=4096, num_validation_samples=250)
        """ generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function = dl.get_all_sensor_sequences, data_gather_function = dl.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs ,max_queue_length=4096, num_validation_samples=250) """
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()
    
    return generator

def get_comment():
    comment = "Tried lower learning rate and slightly different dropout and fc configs."
    return comment

if __name__ == "__main__":
    #torch.cuda.set_device(0)
    #generator = create_dataGenerator_IMG()
    #generator = create_dataGenerator_index_sequence()
    #generator = create_dataGenerator_single_state()
    print(">>> INFO: Generating Model")
    model = ERFH5_PressureSequence_Model()
    print(">>> INFO: Generating Generator")
    generator = create_dataGenerator_pressure_sequence()
    print(">>> INFO: Model to GPU")
    model = nn.DataParallel(model).to('cuda:0')
    print(">>> INFO: Generating Trainer")
    train_wrapper = Master_Trainer(model, generator, loss_criterion=torch.nn.BCELoss(), comment=get_comment(), savepath=savepath)
    print(">>> INFO: The Training Will Start Shortly")
    train_wrapper.start_training()
    train_wrapper.saveModel()
    print("model saved")
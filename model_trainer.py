import erfh5_pipeline as pipeline
from Generic_Trainer import Master_Trainer
import data_loaders as dl
import torch
from torch import nn
from erfh5_RNN_models import ERFH5_RNN
from erfh5_autoencoder import erfh5_Distributed_Autoencoder
from erfh5_ConvModel import erfh5_Conv3d

batchsize = 16
epochs = 80


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



if __name__ == "__main__":
    #torch.cuda.set_device(0)
    generator = create_dataGenerator_IMG()
    #generator = create_dataGenerator_index_sequence()
    #generator = create_dataGenerator_single_state()
    #model = ERFH5_RNN(69366, 3072, batchsize)
    model = erfh5_Conv3d(21)
    model = nn.DataParallel(model).to('cuda:0')
    #model = erfh5_Distributed_Autoencoder()

    train_wrapper = Master_Trainer(model, generator)

    train_wrapper.start_training()
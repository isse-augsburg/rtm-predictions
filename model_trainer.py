
from Pipeline import erfh5_pipeline as pipeline, data_loaders as dl, data_loader_sensor as dls, data_loaders_IMG as dli, \
    data_gather as dg
from Trainer.Generic_Trainer import Master_Trainer
from Trainer.evaluation import Binary_Classification_Evaluator
import torch
import traceback
from torch import nn
from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model


batchsize = 1
max_Q_len = 512
epochs = 70
#path = ['/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/']
path = ['/cfs/share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/', '/cfs/share/data/RTM/Lautern/output/with_shapes/2019-04-23_10-23-20_200p']
#path = ['/cfs/share/data/RTM/Lautern/output/with_shapes']


def create_dataGenerator_pressure_flowfront():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path=path, num_validation_samples=100,
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
        generator = pipeline.ERFH5_DataGenerator(data_path=path, num_validation_samples=10,
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
        generator = pipeline.ERFH5_DataGenerator(data_path="/cfs/home/s/c/schroeni/Data/Images",
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
        generator = pipeline.ERFH5_DataGenerator(data_path="/cfs/home/s/c/schroeni/Git/tu-kaiserslautern-data/Images",
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
        generator = pipeline.ERFH5_DataGenerator(data_path='/cfs/share/data/RTM/Lautern/clean_erfh5/',
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
            path, data_processing_function=dls.get_sensordata_and_filling_percentage,
            data_gather_function=dg.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs, max_queue_length=max_Q_len, num_validation_samples=70)
        """ generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function = dl.get_all_sensor_sequences, data_gather_function = dl.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs ,max_queue_length=4096, num_validation_samples=250) """
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()

    return generator


def get_comment():
    comment = "Sensor values are now correctyl scaled"
    return comment


if __name__ == "__main__":
    # torch.cuda.set_device(0)
    # generator = create_dataGenerator_IMG()
    # generator = create_dataGenerator_index_sequence()
    # generator = create_dataGenerator_single_state()
    # model = ERFH5_PressureSequence_Model()
    # generator = create_dataGenerator_pressure_sequence()

    print(">>> INFO: Generating Model")
    model = ERFH5_PressureSequence_Model()
    print(">>> INFO: Generating Generator")
    generator = create_dataGenerator_pressure_sequence()
    print(">>> INFO: Model to GPU")
    model = nn.DataParallel(model).to('cuda:0')
    print(">>> INFO: Generating Trainer")
    train_wrapper = Master_Trainer(model, generator, loss_criterion=torch.nn.BCELoss(), comment=get_comment(),
                                   savepath='/cfs/home/l/o/lodesluk/models/crnn_1505_1045.pt', learning_rate=0.0001, classification_evaluator=Binary_Classification_Evaluator())
    print(">>> INFO: The Training Will Start Shortly")

    train_wrapper.start_training()
    train_wrapper.save_model()
    print("Model saved.")

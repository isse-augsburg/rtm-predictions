import torch
from Pipeline import erfh5_pipeline as pipeline, data_loaders as dl, data_gather as dg, data_loader_sensor  as dls
from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from collections import OrderedDict
from Debugging import erfh5_img_Representations as img_reps

# path = ['/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/1_solved_simulations/20_auto_solver_inputs/']
path = ['Y:/data/RTM/Lautern/1_solved_simulations/20_auto_solver_inputs']
# model_path = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/l/o/lodesluk/models/crnn_weird.pt'
model_path = 'Z:/models/crnn_weird.pt'


def create_dataGenerator_pressure_sequence():
    try:
        generator = pipeline.ERFH5DataGenerator(
            path, data_processing_function=dls.get_sensordata_and_filling_percentage,
            data_gather_function=dg.get_filelist_within_folder,
            batch_size=1, epochs=1, max_queue_length=256, num_validation_samples=2)
        """ generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function = dl.get_all_sensor_sequences, data_gather_function = dl.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs ,max_queue_length=4096, num_validation_samples=250) """
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()

    return generator

if __name__ == "__main__":
    model = ERFH5_PressureSequence_Model()
    generator = create_dataGenerator_pressure_sequence()
    state_dict = torch.load(model_path, map_location='cpu') 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params   
    model.load_state_dict(new_state_dict)

    """ class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            image_modules = list(model.cnn.children())[:-1] #all layer expect last layer
            self.modelA = torch.nn.Sequential(*image_modules)
            
        def forward(self, image):
            a = self.modelA(image)
            x = torch.nn.functional.sigmoid(a)
            return x

    model = MyModel() """
    print(">>>INFO: Loaded model")

    all_paths = dg.get_filelist_within_folder(path)

    for f in all_paths:
        batch_data, batch_labels = generator.__next__()
        instance = dls.get_sensordata_and_filling_percentage(f)
        tensor_instances = list()

        for i in instance:
            data, label = torch.FloatTensor(i[0]), torch.FloatTensor(i[1])
            tensor_instances.append((data, label))

        data = [t[0] for t in tensor_instances]
        labels = [t[1] for t in tensor_instances]
        data = torch.stack(data)
        labels = torch.stack(labels)

        output = model(data)
        if output[0][0] <= 0.99:
            print("Filename: ", f)
            print("Output: ", output, "|| Label: ", labels)
            print("-------------------------------------------")
            img_reps.create_images_for_file(f, main_folder="Z:/CRNN_Debug_2/")
    print("end")


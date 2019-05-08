import torch 
import torch.nn as nn 
import Generic_Trainer 
import erfh5_pipeline as pipeline 
import data_loaders as dl 
from erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from Generic_Trainer import Master_Trainer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = ['/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/1_solved_simulations/20_auto_solver_inputs/'] 
model_path = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/l/o/lodesluk/models/crnn1.pt'

def create_dataGenerator_pressure_sequence(): 
    try: 
        generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function = dl.get_sensordata_and_filling_percentage, data_gather_function = dl.get_filelist_within_folder,
            batch_size=2, epochs=1 ,max_queue_length=256, num_validation_samples=2)
        """ generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function = dl.get_all_sensor_sequences, data_gather_function = dl.get_filelist_within_folder,
            batch_size=batchsize, epochs=epochs ,max_queue_length=4096, num_validation_samples=250) """
    except Exception as e:
        print(">>>ERROR: Fatal Error:", e)
        exit()
    
    return generator

print(">>> INFO: Generating Model")
model = ERFH5_PressureSequence_Model()
print(">>> INFO: Generating Generator")
generator = create_dataGenerator_pressure_sequence()
train_wrapper = Master_Trainer(model, generator, loss_criterion=torch.nn.BCELoss(), savepath= model_path)

train_wrapper.load_model()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        image_modules = list(train_wrapper.model.cnn.children())[:-1] #all layer expect last layer
        self.modelA = torch.nn.Sequential(*image_modules)
        
    def forward(self, image):
        a = self.modelA(image)
        x = torch.nn.functional.sigmoid(a)
        return x

model = MyModel()
print("loaded model")
x_image = torch.zeros(1, 1, 224, 360)
out = model(x_image)
out = out.detach().numpy()
out = np.squeeze(out)

print(out)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(out[:,0], out[:,1], out[:,2])
plt.show()
### Short Guide to the RTM-Predictions Data-Pipeline (regular use):

Note: A complete documentation can be found in `/cfs/share/cache/website/html/index.html` or http://137.250.170.59:8000/ or
https://rtm-predictions.readthedocs.io/en/latest/index.html (which might be not as up to date es the others).

* Most of the work is done by the `ModelTrainer` class (defined in `Trainer.ModelTrainer.py`)
* `ModelTrainer` is the generic base class which takes all necessary parameters for any kind of training
* The training process is configured and started from an own dedicated script
* Examples for these dedicated scripts can be found in the model_trainer_*.py files in the root directory 

Basic principles of ModelTrainer:
* Data processing is currently done by the 'LoopingDataGenerator' (defined in `Pipeline.torch_datagenerator.py`) 
* `LoopingDataGenerator` takes a list of file paths as base paths
* The base paths are searched for .erfh5 files using the `data_gather_function` passed to the ModelTrainer 
* After gathering the .erfh5 files, the data from these is processed using the `data_processing_function` passed to the ModelTrainer. An example for processing is the extraction of all pressure sensor values 
* Additional work such as creating batches and shuffling data is done automatically
* The training process is implemented in ModelTrainer. This includes validation steps during training and testing on a dedicated test set after training
 
Steps for using the ModelTrainer in your script: 
* For data processing you need two functions: 
    1. `data_gather_function`, a function for collecting the paths to the files from a root directory 
    2. `data_processing_function,` a function that extracts the data from the collected filepaths. **Must** return data in following format: `[(instance_1, label_1), ... , (instance_n, label_n)]`
    (examples for both functions can be found in the `Pipeline.data_loader_*.py` and `Pipeline.data_gather.py` files)
* Define a PyTorch model 
* Instantiate the ModelTrainer: `mt = ModelTrainer( ... )`. Pass all necessary arguments for your task. Important: You have to pass your model using `lambda: YourModel()`
* Train your model using `mt.start_training()`. No additional parameters need to be passed if you have configured the ModelTrainer correctly
* Testing using a dedicated test set can be done using `mt.inference_on_test_set( ... )`
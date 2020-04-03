## General:
To run the code, either use the environment.yaml with anaconda or even easier use Docker:
```
cd Docker
docker build -t "pytorch_extended:20.02"
docker run ...
```

## Rerun experiments from FlowFrontNet paper:

1. Download the data from here:
https://figshare.com/s/6d8ebc90e0e820b7f08f.
There are two big Zip files: SensorToDryspot, SensorToFlowFront.
To recreate the experiments from the paper, we need both.

The SensorToDryspot dataset can be used for:
* DenseNet baseline
* Finetuning FlowFrontNet with a pretrained Deconv / Conv 

The SensorToFlowFront dataset can be used to train the Deconv / Conv Network to produce FlowFrontImages

2. Unzip those files and put it in the root directory of this repository

3. Trainings:
* Start the following script for 1140 sensors to flowfront:
`python3 -u ModelTrainerScripts.model_trainer_sensor_1140_to_flow.py --demo`

* To use the fine-tuned model for binary classification:
`python3 -u ModelTrainerScripts.model_trainer_sensor_1140_to_dryspot.py --demo`

* For the baseline, run:
`python3 -u ModelTrainerScripts.model_trainer_sensor_1140_dryspot_end_to_end_dense.py --demo`

4. Evaluation:
* Start the following script for 1140 sensors to flowfront:
`python3 -u ModelTrainerScripts.model_trainer_sensor_1140_to_flow.py --demo --eval`

* To use the fine-tuned model for binary classification:
`python3 -u ModelTrainerScripts.model_trainer_sensor_1140_to_dryspot.py --demo --eval`

* For the baseline, run:
`python3 -u ModelTrainerScripts.model_trainer_sensor_1140_dryspot_end_to_end_dense.py --demo --eval`

Caution: New Folders will be created in the current directory.

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



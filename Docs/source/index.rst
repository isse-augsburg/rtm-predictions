TU Kaiserslautern
======================
Project for working with simulated data.

.. note:: 
    The Pipeline is still work in progress and may has unexpected behaviour and/or some code flaws

Tutorial for training models
----------------------------

* Most of the work is done by the `Master_Trainer` class (defined in `Trainer.Generic_Trainer.py`)
* Currently, `model_trainer.py` is used for training
* Training can be started using the `run_model_trainer.sh` batch file (can be found in `/cfs/home/l/o/lodesluk/slurm/`, which calls `model_trainer.py` 

**Steps for using the Master_Trainer**: 

* Create a ERFH5_DataGenerator (defined in `Pipeline.erfh5_pipeline.py`), which needs

1. a function for collecting the paths to the files from a root directory

2. a function that extracts the data from the collected filepaths. **Must** return data in following format: `[(instance_1, label_1), ... , (instance_n, label_n)].` (examples for botch functions can be found in the `Pipeline.data_loader_*.py` and `Pipeline.data_gather.py` files)

* Define a PyTorch model

* Create the Master_Trainer: `train_wrapper = Master_Trainer(model, generator)`

* Start training: `train_wrapper.start_training()`

* There are many more optional arguments for Master_Trainer and ERFH5_DataGenerator 

Modules
======================

.. toctree::
   :maxdepth: 4

   Debugging
   Legacy
   Models
   Pipeline
   Simulation
   Trainer
   model_trainer
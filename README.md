## General:
To run the code, either use the environment.yaml with anaconda or even easier use Docker:
```
cd Docker
docker build -t "pytorch_extended:20.02"
docker run ...
```

## Rerun experiments from FlowFrontNet paper:

1.  Download the data and checkpoints from here:
https://figshare.com/s/dde2f78958173c23aee4.
There are two big Zip files: SensorToDryspot, SensorToFlowFront.
To recreate the experiments from the paper, we need both.

2. The SensorToDryspot dataset can be used for:
    * Feedforward baseline
    * Finetuning FlowFrontNet with a pretrained Deconv / Conv 

The SensorToFlowFront dataset can be used to train the Deconv / Conv Network to produce FlowFrontImages

2. Unzip those files in a certain `data_path`

3. Start Trainings:
    * Start the following script for 1140 sensors to flowfront:
    `python3 -u ModelTrainerScripts.model_trainer_sensor_1140_to_flow.py --demo data_path`

    * To use the fine-tuned model for binary classification:
    `python3 -u ModelTrainerScripts.model_trainer_sensor_1140_to_dryspot.py --demo data_path --checkpoint_path checkpoint_path`

    * For the baseline, run:
    `python3 -u ModelTrainerScripts.model_trainer_sensor_1140_dryspot_end_to_end_dense.py --demo`

4. Evaluation:
    * Start the following script for 1140 sensors to flowfront:
    `python3 -u ModelTrainerScripts.model_trainer_sensor_1140_to_flow.py --demo data_path --eval --checkpoint_path checkpoint_path`
    
    * To use the fine-tuned model for binary classification:
    `python3 -u ModelTrainerScripts.model_trainer_sensor_1140_to_dryspot.py --demo data_path --eval --checkpoint_path checkpoint_path`
    
    * For the baseline, run:
    `python3 -u ModelTrainerScripts.model_trainer_sensor_1140_dryspot_end_to_end_dense.py --demo data_path --eval --checkpoint_path checkpoint_path`

Caution: New Folders will be created in the directory of the Datasets, corresponding to the task: SensorToFlowFront or SensorToDryspot.



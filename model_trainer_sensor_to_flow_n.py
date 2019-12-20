from pathlib import Path

import torch

import Resources.resources_for_training as r
from Models.erfh5_DeconvModel import DeconvModelEfficientBn
from Pipeline.Utils.looping_strategies import ComplexListLoopingStrategy
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import get_sensordata_and_flowfront_149x117_ignore_useless
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params
from general_model_trainer import ModelTrainer

if __name__ == "__main__":
    args = read_cmd_params()

    num_samples_runs = 1860000
    batch_size = 1024
    m = ModelTrainer(DeconvModelEfficientBn(),
                     r.get_data_paths(),
                     r.save_path,
                     load_datasets_path=r.datasets_dryspots,
                     cache_path=r.cache_path,
                     batch_size=batch_size,
                     eval_freq=int(num_samples_runs / batch_size),
                     train_print_freq=10,
                     epochs=1000,
                     num_workers=75,
                     num_validation_samples=8192,
                     num_test_samples=8192,
                     data_processing_function=get_sensordata_and_flowfront_149x117_ignore_useless,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     looping_strategy=ComplexListLoopingStrategy(batch_size)
                     )

    if not args.eval:
        m.run_training(
            loss_criterion=torch.nn.MSELoss(),
            learning_rate=0.0001,
            calc_metrics=False,
            classification_evaluator=SensorToFlowfrontEvaluator()
        )
    else:
        m.inference_on_test_set(Path(args.eval_path),
                                SensorToFlowfrontEvaluator(Path(args.eval_path) / "eval_on_test_set",
                                                           skip_images=False))

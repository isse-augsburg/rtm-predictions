from pathlib import Path

import torch

import Resources.resources_for_training as r
from Models.erfh5_ConvModel import SensorDeconvToDryspotEfficient
from Pipeline.Utils.looping_strategies import ComplexListLoopingStrategy
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import get_sensor_bool_dryspot_ignore_useless
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params
from general_model_trainer import ModelTrainer

if __name__ == "__main__":
    args = read_cmd_params()

    num_samples_runs = 1860000
    batch_size = 1024
    loss = torch.nn.BCELoss()
    m = ModelTrainer(SensorDeconvToDryspotEfficient(pretrained="all_weights",
                                                    checkpoint_path=r.chkp_S1140_to_ds_frozen_deeper_convnet2,
                                                    freeze_nlayers=8),
                     r.get_data_paths_new_test_set(),
                     r.save_path,
                     load_datasets_path=None,
                     cache_path=r.cache_path,
                     batch_size=batch_size,
                     eval_freq=int(num_samples_runs / batch_size),
                     train_print_freq=10,
                     epochs=1000,
                     num_workers=75,
                     num_validation_samples=0,
                     num_test_samples=131072,
                     data_processing_function=get_sensor_bool_dryspot_ignore_useless,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     looping_strategy=ComplexListLoopingStrategy(batch_size)
                     )

    if not args.eval:
        m.run_training(
            loss_criterion=loss,
            learning_rate=0.0001,
            calc_metrics=False,
            classification_evaluator=BinaryClassificationEvaluator()
        )
    else:
        m.inference_on_test_set(Path(args.eval_path),
                                BinaryClassificationEvaluator(Path(args.eval_path) / "eval_on_test_set",
                                                              # TODO fix Image creation when handling sensor input
                                                              #  reshape etc.
                                                              skip_images=True,
                                                              with_text_overlay=True),
                                loss_criterion=loss)

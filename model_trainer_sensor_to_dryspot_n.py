import torch

import Resources.resources_for_training as r
from Models.erfh5_ConvModel import SensorDeconvToDryspot
from Pipeline.Utils.looping_strategies import ComplexListLoopingStrategy
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import get_sensor_bool_dryspot
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params
from general_model_trainer import ModelTrainer

if __name__ == "__main__":
    args = read_cmd_params()

    num_samples_runs = 2000000
    batch_size = 512
    m = ModelTrainer(SensorDeconvToDryspot(),
                     r.get_data_paths(),
                     r.save_path,
                     load_datasets_path=r.datasets_dryspots,
                     cache_path=r.cache_path,
                     batch_size=batch_size,
                     eval_freq=int(num_samples_runs / batch_size),
                     train_print_freq=10,
                     epochs=1000,
                     num_workers=20,
                     num_validation_samples=8192,
                     num_test_samples=8192,
                     data_processing_function=get_sensor_bool_dryspot,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     looping_strategy=ComplexListLoopingStrategy(batch_size)
                     )

    if not args.eval:
        m.run_training(
            loss_criterion=torch.nn.BCELoss(),
            learning_rate=0.0001,
            calc_metrics=False,
            classification_evaluator=BinaryClassificationEvaluator()
        )
    else:
        m.inference_on_test_set(args.eval_path, BinaryClassificationEvaluator(args.eval_path / "eval_on_test_set",
                                                                              skip_images=False,
                                                                              with_text_overlay=True))

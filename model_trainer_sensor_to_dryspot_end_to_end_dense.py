from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_fullyConnected import S1140DryspotModelFCWide
from Pipeline.TorchDataGeneratorUtils.looping_strategies import ComplexListLoopingStrategy
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import get_sensor_bool_dryspot_ignore_useless
from Trainer.GenericTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    batch_size = 131072
    m = ModelTrainer(lambda: S1140DryspotModelFCWide(),
                     data_source_paths=r.get_data_paths_base_0(),
                     save_path=r.save_path,
                     load_datasets_path=r.datasets_dryspots,
                     cache_path=r.cache_path,
                     batch_size=batch_size,
                     epochs=100,
                     num_workers=10,
                     num_validation_samples=8192,
                     num_test_samples=8192,
                     data_processing_function=get_sensor_bool_dryspot_ignore_useless,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     looping_strategy=ComplexListLoopingStrategy(batch_size),
                     loss_criterion=torch.nn.BCELoss(),
                     classification_evaluator=BinaryClassificationEvaluator()
                     )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(output_path=Path(args.eval_path),
                                checkpoint_path=Path(args.checkpoint_path),
                                classification_evaluator=BinaryClassificationEvaluator(Path(args.eval_path) /
                                                                                       "eval_on_test_set",
                                                                                       skip_images=True,
                                                                                       with_text_overlay=True))

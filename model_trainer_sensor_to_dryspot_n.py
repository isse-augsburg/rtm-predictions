from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_ConvModel import SensorDeconvToDryspotEfficient
from Pipeline.TorchDataGeneratorUtils.looping_strategies import ComplexListLoopingStrategy
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import get_sensor_bool_dryspot_ignore_useless
from Trainer.GenericTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    num_samples_runs = 1937571
    batch_size = 1024
    m = ModelTrainer(lambda: SensorDeconvToDryspotEfficient(pretrained="deconv_weights",
                                                            checkpoint_path=r.chkp_1140_transfered_dry_spot,
                                                            freeze_nlayers=8),
                     r.get_data_paths(),
                     r.save_path,
                     load_datasets_path=r.datasets_dryspots,
                     cache_path=r.cache_path,
                     batch_size=batch_size,
                     train_print_frequency=10,
                     epochs=1000,
                     num_workers=75,
                     num_validation_samples=8192,
                     num_test_samples=8192,
                     data_processing_function=get_sensor_bool_dryspot_ignore_useless,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     looping_strategy=ComplexListLoopingStrategy(batch_size),
                     loss_criterion=torch.nn.BCELoss(),
                     learning_rate=0.0001,
                     classification_evaluator=BinaryClassificationEvaluator()
                     )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(Path(args.eval_path), Path(args.checkpoint_path),
                                BinaryClassificationEvaluator(Path(args.eval_path) / "eval_on_test_set",
                                                              # TODO fix Image creation when handling sensor input
                                                              #  reshape etc.
                                                              skip_images=True,
                                                              with_text_overlay=True))

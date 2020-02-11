from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_ConvModel import S20DeconvToDrySpotEff
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.GenericTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    dlds = DataloaderDryspots()
    m = ModelTrainer(lambda: S20DeconvToDrySpotEff(pretrained="deconv_weights",
                                                   checkpoint_path=r.chkp_20_sensors_to_ff,
                                                   freeze_nlayers=5),
                     r.get_data_paths(),
                     r.save_path,
                     load_datasets_path=r.datasets_dryspots,
                     cache_path=r.cache_path,
                     batch_size=8192,
                     train_print_frequency=10,
                     epochs=1000,
                     num_workers=75,
                     num_validation_samples=8192,
                     num_test_samples=8192,
                     data_processing_function=dlds.get_sensor_bool_dryspot,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     loss_criterion=torch.nn.BCELoss(),
                     optimizer_function=lambda params: torch.optim.Adam(params, lr=0.00001),
                     classification_evaluator=BinaryClassificationEvaluator()
                     )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(output_path=Path(args.eval_path),
                                checkpoint_path=Path(args.checkpoint_path),
                                # TODO fix Image creation when handling sensor input
                                #  reshape etc.
                                classification_evaluator=BinaryClassificationEvaluator(Path(args.eval_path) /
                                                                                       "eval_on_test_set",
                                                                                       skip_images=True,
                                                                                       with_text_overlay=True))

from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_ConvModel import S20DeconvToDrySpotEff2
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.GenericTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    dlds = DataloaderDryspots(sensor_indizes=((1, 8), (1, 8)))
    m = ModelTrainer(lambda: S20DeconvToDrySpotEff2(pretrained="deconv_weights",
                                                    checkpoint_path=r.chkp_S20_to_ff_basepr_0,
                                                    freeze_nlayers=5),
                     data_source_paths=r.get_data_paths_base_0(),
                     save_path=r.save_path,
                     load_datasets_path=r.datasets_dryspots,
                     cache_path=r.cache_path,
                     batch_size=4096,
                     train_print_frequency=100,
                     epochs=1000,
                     num_workers=40,
                     num_validation_samples=131072,
                     num_test_samples=1048576,
                     data_processing_function=dlds.get_sensor_bool_dryspot,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     loss_criterion=torch.nn.BCELoss(),
                     optimizer_function=lambda params: torch.optim.AdamW(params, lr=0.0001),
                     classification_evaluator_function=lambda sw: BinaryClassificationEvaluator(summary_writer=sw)
                     )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(output_path=Path(args.eval_path),
                                checkpoint_path=Path(args.checkpoint_path),
                                # TODO fix Image creation when handling sensor input
                                #  reshape etc.
                                classification_evaluator_function=BinaryClassificationEvaluator(Path(args.eval_path) /
                                                                                                "eval_on_test_set",
                                                                                                skip_images=True,
                                                                                                with_text_overlay=True))

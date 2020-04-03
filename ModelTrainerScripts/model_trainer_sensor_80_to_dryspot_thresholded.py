from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_ConvModel import S80Deconv2ToDrySpotEff
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    dl = DataloaderDryspots(sensor_indizes=((1, 4), (1, 4)))

    # def get_sampler(data_source):
    #     return RandomOverSampler(data_source, multiply_by=2)

    m = ModelTrainer(
        lambda: S80Deconv2ToDrySpotEff(pretrained="deconv_weights",
                                       checkpoint_path=r.chkp_S80_to_ff2,
                                       freeze_nlayers=9,
                                       round_at=0.8),
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path,
        load_datasets_path=r.datasets_dryspots,
        cache_path=r.cache_path,
        batch_size=8192,
        train_print_frequency=100,
        epochs=1000,
        num_workers=75,
        num_validation_samples=131072,
        num_test_samples=1048576,
        data_processing_function=dl.get_sensor_bool_dryspot,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=0.0001),
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(summary_writer=summary_writer),
        # lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.8),
        # sampler=get_sampler
    )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval_path),
            Path(args.checkpoint_path),
            lambda summary_writer: BinaryClassificationEvaluator(
                Path(args.eval_path) / "eval_on_test_set",
            ),
        )

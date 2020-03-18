from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_DeconvModel import S80DeconvModelEfficient2
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImages
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    dl = DataloaderImages(image_size=(112, 96),
                          sensor_indizes=((1, 4), (1, 4)),
                          divide_by_100k=False)

    m = ModelTrainer(
        lambda: S80DeconvModelEfficient2(),
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path,
        load_datasets_path=r.datasets_dryspots,
        cache_path=r.cache_path,
        batch_size=4096,
        train_print_frequency=100,
        epochs=1000,
        num_workers=75,
        num_validation_samples=131072,
        num_test_samples=1048576,
        data_processing_function=dl.get_sensordata_and_flowfront,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=0.001),
        classification_evaluator_function=lambda summary_writer:
        SensorToFlowfrontEvaluator(summary_writer=summary_writer),
        use_mixed_precision=True
    )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval_path),
            Path(args.checkpoint_path),
            lambda summary_writer: SensorToFlowfrontEvaluator(
                Path(args.eval_path) / "eval_on_test_set",
                skip_images=False,
                sensors_shape=(10, 8),
                print_n_images=5000
            ),
        )

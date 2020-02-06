from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_DeconvModel import DeconvModelEfficient
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImages
from Trainer.GenericTrainer import ModelTrainer
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    batch_size = 2048
    dl = DataloaderImages((149, 117))
    m = ModelTrainer(
        lambda: DeconvModelEfficient(),
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path,
        load_datasets_path=None,
        cache_path=r.cache_path,
        batch_size=batch_size,
        train_print_frequency=10,
        epochs=1000,
        num_workers=75,
        num_validation_samples=131072 // 2,
        num_test_samples=1048576 // 2,
        data_processing_function=dl.get_sensordata_and_flowfront,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=0.0001),
        classification_evaluator=SensorToFlowfrontEvaluator(),
    )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval_path),
            Path(args.checkpoint_path),
            SensorToFlowfrontEvaluator(
                Path(args.eval_path) / "eval_on_test_set", skip_images=False
            ),
        )

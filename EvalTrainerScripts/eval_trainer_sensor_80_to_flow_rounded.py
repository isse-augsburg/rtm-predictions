import torch

import Resources.training as r
from Models.erfh5_DeconvModel import S80DeconvModelEfficient2
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImages
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import SensorToFlowfrontEvaluator

if __name__ == "__main__":
    dl = DataloaderImages(image_size=(112, 96),
                          sensor_indizes=((1, 4), (1, 4)))

    checkpoint_p = r.chkp_S80_to_ff2
    adv_output_dir = checkpoint_p.parent / "advanced_eval"

    m = ModelTrainer(
        lambda: S80DeconvModelEfficient2(pretrained="all",
                                         freeze_nlayers=9,
                                         checkpoint_path=checkpoint_p,
                                         round_at=.8),
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path,
        load_datasets_path=r.datasets_dryspots,
        cache_path=r.cache_path,
        batch_size=2048,
        train_print_frequency=100,
        epochs=1000,
        num_workers=75,
        num_validation_samples=131072,
        num_test_samples=1048576,
        data_processing_function=dl.get_sensordata_and_flowfront,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=0.0001),
        classification_evaluator_function=lambda summary_writer:
        SensorToFlowfrontEvaluator(summary_writer=summary_writer),
    )

    output_path = r.chkp_S80_to_ff2.parent
    m.inference_on_test_set(
        output_path / "eval_on_test_set_rounded.5",
        r.chkp_S80_to_ff2,
        lambda summary_writer: SensorToFlowfrontEvaluator(
            output_path / "eval_on_test_set_rounded.5",
            skip_images=False,
            sensors_shape=(10, 8),
            print_n_images=5000
        ),
    )

import torch

import Resources.training as r
from Models.erfh5_DeconvModel import DeconvModelEfficient
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImages
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import SensorToFlowfrontEvaluator

if __name__ == "__main__":
    dl = DataloaderImages((149, 117))

    checkpoint_p = r.chkp_S1140_to_ff_0_basepr
    adv_output_dir = checkpoint_p.parent / "advanced_eval"

    m = ModelTrainer(
        lambda: DeconvModelEfficient(),
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path,
        load_datasets_path=r.datasets_dryspots,
        cache_path=r.cache_path,
        batch_size=2048,
        train_print_frequency=10,
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

    adv_output_dir.mkdir(exist_ok=True)
    m.inference_on_test_set(
        output_path=adv_output_dir,
        checkpoint_path=checkpoint_p,
        classification_evaluator_function=lambda summary_writer:
        SensorToFlowfrontEvaluator(adv_output_dir,
                                   skip_images=False,
                                   print_n_images=5000)
    )

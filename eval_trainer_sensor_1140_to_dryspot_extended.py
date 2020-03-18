import pickle

import torch

import Resources.training as r
from Models.erfh5_ConvModel import SensorDeconvToDryspotEfficient2
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    dlds = DataloaderDryspots()
    m = ModelTrainer(lambda: SensorDeconvToDryspotEfficient2(pretrained="all",
                                                             checkpoint_path=r.chkp_S1140_to_ds_0_basepr_frozen,
                                                             freeze_nlayers=8),
                     data_source_paths=r.get_data_paths_base_0(),
                     save_path=r.save_path,
                     load_datasets_path=r.datasets_dryspots,
                     cache_path=r.cache_path,
                     batch_size=1024,
                     train_print_frequency=100,
                     epochs=1000,
                     num_workers=75,
                     num_validation_samples=131072,
                     num_test_samples=1048576,
                     data_processing_function=dlds.get_sensor_bool_dryspot,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     loss_criterion=torch.nn.BCELoss(),
                     optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
                     classification_evaluator_function=lambda summary_writer:
                     BinaryClassificationEvaluator(summary_writer=summary_writer)
                     )

    m.inference_on_test_set(
        output_path=r.dir_S1140_to_ds / "advanced_eval",
        checkpoint_path=r.chkp_S1140_to_ds_0_basepr_frozen,
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(r.dir_S1140_to_ds / "advanced_eval",
                                      skip_images=True,
                                      with_text_overlay=True)
    )
    with open(r.dir_S1140_to_ds / "advanced_eval/predictions_per_run.p", "wb") as f:
        pickle.dump(m.classification_evaluator.origin_tracker, f)

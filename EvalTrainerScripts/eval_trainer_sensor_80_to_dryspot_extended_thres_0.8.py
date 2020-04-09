import torch

import Resources.training as r
from Models.erfh5_ConvModel import S80Deconv2ToDrySpotEff
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.eval_utils import run_eval_w_binary_classificator

if __name__ == "__main__":
    dl = DataloaderDryspots(sensor_indizes=((1, 4), (1, 4)),
                            aux_info=True)

    checkpoint_p = r.chkp_S80_to_ds_thres
    adv_output_dir = checkpoint_p.parent / "advanced_eval"
    m = ModelTrainer(
        lambda: S80Deconv2ToDrySpotEff(pretrained="all",
                                       checkpoint_path=checkpoint_p,
                                       freeze_nlayers=9,
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
        data_processing_function=dl.get_sensor_bool_dryspot,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=0.0001),
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(summary_writer=summary_writer),
        caching_torch=False
    )

    run_eval_w_binary_classificator(adv_output_dir, m, checkpoint_p)

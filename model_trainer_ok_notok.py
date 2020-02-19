import torch

import Resources.training as r
from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from Pipeline import data_loader_sensor as dls, data_gather as dg
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator

if __name__ == "__main__":
    data_source_paths = [r.data_root / "2019-07-24_16-32-40_5000p"]
    save_path = r.save_path
    cache_path = r.cache_path

    trainer = ModelTrainer(
        lambda: ERFH5_PressureSequence_Model(),
        data_source_paths,
        save_path,
        None,
        epochs=2,
        data_gather_function=dg.get_filelist_within_folder,
        data_processing_function=dls.sensorgrid_simulationsuccess,
        loss_criterion=torch.nn.BCELoss(),
        classification_evaluator_function=BinaryClassificationEvaluator(),
    )
    trainer.start_training()
    print("training finished")

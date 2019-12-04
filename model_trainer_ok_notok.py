from general_model_trainer import ModelTrainer
import torch

from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from Pipeline import data_loader_sensor as dls, data_gather as dg
from Trainer.evaluation import BinaryClassificationEvaluator
import resources_for_training as r


def get_comment():
    return "No comment"


def create_model_trainer(
    data_source_paths,
    save_path,
    load_datasets_path=None,
    cache_path=None,
    batch_size=1,
    max_queue_length=4,
    eval_freq=2,
    train_print_freq=2,
    epochs=10,
    num_workers=10,
    num_validation_samples=10,
    num_test_samples=10,
    data_processing_function=dls.sensorgrid_simulationsuccess,
    data_gather_function=dg.get_filelist_within_folder,
):

    mt = ModelTrainer(
        data_source_paths,
        save_path,
        load_datasets_path,
        cache_path,
        batch_size,
        max_queue_length,
        eval_freq,
        train_print_freq,
        epochs,
        num_workers,
        num_validation_samples,
        num_test_samples,
        data_processing_function,
        data_gather_function,
        model=ERFH5_PressureSequence_Model(),
    )

    return mt


def run_training(
    model_trainer,
    comment=None,
    loss_criterion=torch.nn.BCELoss(),
    learning_rate=0.0001,
    calc_metrics=False,
    classification_evaluator=BinaryClassificationEvaluator(),
):

    model_trainer.run_training(
        get_comment(),
        loss_criterion,
        learning_rate,
        calc_metrics,
        classification_evaluator,
    )


if __name__ == "__main__":
    data_source_paths = [r.data_root]
    save_path = r.save_path
    cache_path = r.cache_path

    model_trainer = create_model_trainer(
        data_source_paths=data_source_paths,
        save_path=save_path,
        load_datasets_path=None,
        cache_path=cache_path,
        batch_size=8,
        max_queue_length=32,
        eval_freq=50,
        train_print_freq=10,
        epochs=50,
        num_workers=12,
        num_validation_samples=50,
        num_test_samples=0
    )

    run_training(model_trainer, comment=get_comment())

import getpass
import logging
import math
import pickle
import socket
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from Models.erfh5_DeconvModel import DeconvModel2x
from Pipeline import (
    erfh5_pipeline as pipeline,
    data_loaders_IMG as dli,
    data_gather as dg,
)
from Trainer.GenericTrainer import MasterTrainer
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils import logging_cfg

num_data_points = 31376
initial_timestamp = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

if socket.gethostname() == "swt-dgx1":
    data_root = Path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes")
    batch_size = 256
    eval_freq = math.ceil(num_data_points / batch_size)
    if getpass.getuser() == "stiebesi":
        save_path = Path("/cfs/share/cache/output_simon")
    elif getpass.getuser() == "schroeni":
        save_path = Path("/cfs/share/cache/output_niklas")
    else:
        save_path = Path("/cfs/share/cache/output")
    epochs = 10
    num_workers = 18
    num_validation_samples = 2000
    num_test_samples = 2000
    cache_path = None

elif socket.gethostname() == "swtse130":
    cache_path = Path(r"C:\Users\stiebesi\CACHE")
    data_root = Path(r"X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes")
    batch_size = 1
    eval_freq = 5
    save_path = Path(r"Y:\cache\output_simon")
    epochs = 10
    num_workers = 10
    num_validation_samples = 10
    num_test_samples = 2000

else:
    data_root = Path(
        "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes"
    )
    save_path = Path(
        r"/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache/output_niklas"
    )
    epochs = 2
    num_workers = 4
    num_validation_samples = 10
    num_test_samples = 10
    batch_size = 4
    eval_freq = 5
    cache_path = "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache"

paths = [
    data_root / "2019-07-23_15-38-08_5000p",
    data_root / "2019-07-24_16-32-40_5000p",
    data_root / "2019-07-29_10-45-18_5000p",
    data_root / "2019-08-23_15-10-02_5000p",
    data_root / "2019-08-24_11-51-48_5000p",
    data_root / "2019-08-25_09-16-40_5000p",
    data_root / "2019-08-26_16-59-08_6000p",
]


def create_datagenerator_pressure_flowfront(paths, save_path=None,
                                            test_mode=False):
    try:
        generator = pipeline.ERFH5DataGenerator(
            data_paths=paths,
            num_validation_samples=num_validation_samples,
            num_test_samples=num_test_samples,
            batch_size=batch_size,
            epochs=epochs,
            max_queue_length=8096,
            data_processing_function=dli.get_sensordata_and_flowfront_143x111,
            data_gather_function=dg.get_filelist_within_folder,
            num_workers=num_workers,
            cache_path=cache_path,
            save_path=save_path,
            test_mode=test_mode,
        )
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception("Fatal Error:")
        exit()
    return generator


def get_comment():
    return "Hallo"


def inference_on_test_set(path):
    save_path = path / "eval_on_test_set"
    save_path.mkdir(parents=True, exist_ok=True)
    logging_cfg.apply_logging_config(save_path, eval=True)

    model = DeconvModel2x()
    if socket.gethostname() == "swt-dgx1":
        model = nn.DataParallel(model).to("cuda:0")
    else:
        model = model.to("cuda:0")
    gen = create_datagenerator_pressure_flowfront(paths=[], test_mode=True)
    eval_wrapper = MasterTrainer(
        model,
        gen,
        classification_evaluator=SensorToFlowfrontEvaluator(
            save_path=save_path),
    )
    eval_wrapper.load_checkpoint(path / "checkpoint.pth")

    test_set = pickle.load(open(path / "test_set.p", "rb"))
    if socket.gethostname() == "swtse130":
        win_paths = []
        for e in test_set:
            if e[:4] == "/cfs":
                win_paths.append(Path(e.replace("/cfs/home", "X:")))
        test_set = win_paths
    data_list = []
    full = False
    for p in test_set:
        instance = gen.data_function(p)
        for num, i in enumerate(instance):
            data, label = torch.FloatTensor(i[0]), torch.FloatTensor(i[1])
            data_list.append((data, label))
            if len(data_list) >= num_test_samples:
                full = True
        if full:
            data_list = data_list[:num_test_samples]
            break

    eval_wrapper.eval(data_list, test_mode=True)


def run_training(save_path):
    save_path = save_path / initial_timestamp
    save_path.mkdir(parents=True, exist_ok=True)
    logging_cfg.apply_logging_config(save_path)

    logger = logging.getLogger(__name__)

    logger.info("Generating Generator")
    generator = create_datagenerator_pressure_flowfront(
        paths, save_path, test_mode=False
    )
    logger.info("Generating Model")
    model = DeconvModel2x()
    logger.info("Model to GPU")
    model = nn.DataParallel(model).to("cuda:0")

    train_wrapper = MasterTrainer(
        model,
        generator,
        comment=get_comment(),
        loss_criterion=torch.nn.MSELoss(),
        # loss_criterion=pixel_wise_loss_multi_input_single_label,
        savepath=save_path,
        learning_rate=0.0001,
        calc_metrics=False,
        train_print_frequency=2,
        eval_frequency=eval_freq,
        classification_evaluator=SensorToFlowfrontEvaluator(
            save_path=save_path, halving_factor=True
        ),
    )
    logger.info("The Training Will Start Shortly")

    train_wrapper.start_training()
    logger.info("Model saved.")


if __name__ == "__main__":
    train = False
    if train:
        run_training(save_path)
    else:
        if socket.gethostname() != "swtse130":
            inference_on_test_set(
                Path("/cfs/share/cache/output_simon/2019-08-29_16-45-59")
            )
        else:
            inference_on_test_set(
                Path(r"Y:\cache\output_simon\2019-09-05_18-38-33"))

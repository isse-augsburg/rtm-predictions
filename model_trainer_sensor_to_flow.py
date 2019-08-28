import logging
import math
import socket
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from Models.erfh5_DeconvModel import DeconvModel
from Pipeline import erfh5_pipeline as pipeline, data_loaders_IMG as dli, \
    data_gather as dg
from Trainer.Generic_Trainer import Master_Trainer
from Trainer.evaluation import Sensor_Flowfront_Evaluator

num_data_points = 10371
initial_timestamp = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

if socket.gethostname == 'swt-dgx1':
    data_root = Path('/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes')
    batch_size = 256
    eval_freq = math.ceil(num_data_points / batch_size)
    save_path = Path("/cfs/share/cache/output_niklas")
    epochs = 10
    num_workers = 18
    num_validation_samples = 2000

elif socket.gethostname() == 'swtse130':
    data_root = Path(r'X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes')
    batch_size = 1
    eval_freq = 5
    save_path = Path(r"Y:\cache\output_simon")
    epochs = 10
    num_workers = 10
    num_validation_samples = 10


paths = []
paths.append(data_root / '2019-07-23_15-38-08_5000p')
paths.append(data_root / '2019-07-24_16-32-40_5000p')
paths.append(data_root / '2019-07-29_10-45-18_5000p')
paths.append(data_root / '2019-08-23_15-10-02_5000p')
paths.append(data_root / '2019-08-24_11-51-48_5000p')

cache_path = "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache"


def create_dataGenerator_pressure_flowfront():
    try:
        generator = pipeline.ERFH5_DataGenerator(data_paths=paths, num_validation_samples=num_validation_samples,
                                                 batch_size=batch_size, epochs=epochs, max_queue_length=8096,
                                                 data_processing_function=dli.get_sensordata_and_flowfront,
                                                 data_gather_function=dg.get_filelist_within_folder,
                                                 num_workers=num_workers, cache_path=None)
    except Exception as e:
        logger.error("Fatal Error:", e)
        logging.error("exception ", exc_info=1)
        exit()
    return generator


def get_comment():
    return "Hallo"


if __name__ == "__main__":
    save_path = save_path / initial_timestamp
    save_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=save_path / Path('output.log'),
                        level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Generating Generator")
    generator = create_dataGenerator_pressure_flowfront()
    logger.info("Generating Model")
    model = DeconvModel()
    logger.info("Model to GPU")
    model = nn.DataParallel(model).to('cuda:0')

    train_wrapper = Master_Trainer(model, generator,
                                   comment=get_comment(),
                                   loss_criterion=torch.nn.MSELoss(),
                                   # loss_criterion=pixel_wise_loss_multi_input_single_label,
                                   savepath=save_path,
                                   learning_rate=0.0001,
                                   calc_metrics=False,
                                   train_print_frequency=2,
                                   eval_frequency=eval_freq,
                                   classification_evaluator=Sensor_Flowfront_Evaluator(save_path=save_path))
    logger.info("The Training Will Start Shortly")

    train_wrapper.start_training()
    logger.info("Model saved.")

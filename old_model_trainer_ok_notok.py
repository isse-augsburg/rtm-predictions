import logging
import socket
from datetime import datetime
from pathlib import Path
import getpass
import torch

from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
from Pipeline import erfh5_pipeline as pipeline, data_loader_sensor as dls, \
    data_gather as dg
from Trainer.GenericTrainer import MasterTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils import logging_cfg


def get_comment():
    return "Trying 38x30 sensor grid as input for a conventional CNN. Using subsampled version of sequence."


class SuccessTrainer:
    def __init__(self,
                 data_source_paths,
                 save_path,
                 cache_path=None,
                 batch_size=1,
                 eval_freq=2,
                 epochs=10,
                 num_workers=10,
                 num_validation_samples=10,
                 num_test_samples=10):
        self.initial_timestamp = str(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.cache_path = cache_path
        self.data_source_paths = data_source_paths
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.epochs = epochs
        self.num_workers = num_workers
        self.num_validation_samples = num_validation_samples
        self.num_test_samples = num_test_samples
        self.training_data_generator = None
        self.test_data_generator = None

    def create_datagenerator(self, save_path, test_mode=True):
        try:
            generator = pipeline.ERFH5DataGenerator(
                self.data_source_paths,
                data_processing_function=dls.sensorgrid_simulationsuccess,
                data_gather_function=dg.get_filelist_within_folder,
                batch_size=self.batch_size,
                epochs=self.epochs,
                max_queue_length=256,
                num_validation_samples=self.num_validation_samples,
                num_test_samples=self.num_test_samples,
                cache_path=self.cache_path
            )

        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("Fatal Error:")
            exit()
        return generator

    def run_training(self):
        save_path = self.save_path / self.initial_timestamp
        save_path.mkdir(parents=True, exist_ok=True)
        logging_cfg.apply_logging_config(save_path)

        logger = logging.getLogger(__name__)
        logger.info("Generating Generator")
        self.training_data_generator = self.create_datagenerator(save_path,
                                                                 test_mode=False)

        logger.info("Generating Model")
        model = ERFH5_PressureSequence_Model()
        logger.info("Model to GPU")
        model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        train_wrapper = MasterTrainer(
            model,
            self.training_data_generator,
            comment=get_comment(),
            loss_criterion=torch.nn.BCELoss(),
            savepath=save_path,
            learning_rate=0.0005,
            calc_metrics=False,
            train_print_frequency=50,
            eval_frequency=self.eval_freq,
            classification_evaluator=BinaryClassificationEvaluator(),
        )
        logger.info("The Training Will Start Shortly")

        train_wrapper.start_training()
        logging.shutdown()


if __name__ == "__main__":

    if socket.gethostname() == "swt-dgx1":
        print("On DGX.")
        _cache_path = None
        _data_root = Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes")
        _batch_size = 64
        _eval_freq = 50

        if getpass.getuser() == "lodesluk":
            _save_path = Path("/cfs/share/cache/output_lukas")
        else:
            _save_path = Path("/cfs/share/cache/output")
        _epochs = 50
        _num_workers = 18
        _num_validation_samples = 200
        _num_test_samples = 200

    else:
        _cache_path = Path(
            '/run/user/1001/gvfs/smb-share:server=137.250.170.56,'
            'share=share/cache')
        # _cache_path = None

        _data_root = Path(
            '/run/user/1001/gvfs/smb-share:server=137.250.170.56,'
            'share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes')
        _batch_size = 8
        _eval_freq = 50
        _save_path = Path('/home/lodes/Train_Out')
        _epochs = 10
        _num_workers = 4
        _num_validation_samples = 50
        _num_test_samples = 50

    train = True
    if train:
        _data_source_paths = [
            _data_root / "2019-07-23_15-38-08_5000p",
            _data_root / "2019-07-24_16-32-40_5000p",
            _data_root / "2019-07-29_10-45-18_5000p",
            _data_root / "2019-08-23_15-10-02_5000p",
            _data_root / "2019-08-24_11-51-48_5000p",
            _data_root / "2019-08-25_09-16-40_5000p",
            _data_root / "2019-08-26_16-59-08_6000p",
        ]
    else:
        _data_source_paths = []

    st = SuccessTrainer(cache_path=_cache_path,
                        data_source_paths=_data_source_paths,
                        batch_size=_batch_size,
                        eval_freq=_eval_freq,
                        save_path=_save_path,
                        epochs=_epochs,
                        num_workers=_num_workers,
                        num_validation_samples=_num_validation_samples,
                        num_test_samples=_num_test_samples
                        )

    if train:
        st.run_training()

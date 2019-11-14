import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Pipeline import (
    erfh5_pipeline as pipeline,
    data_loaders_IMG as dli,
    data_gather as dg,
)
from Utils import logging_cfg

data_root = Path(
    "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes"
)
cache_path = "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/cache"
path = data_root / "2019-07-23_15-38-08_5000p"
paths = [path]


def create_datagenerator_pressure_flowfront():
    try:
        generator = pipeline.ERFH5DataGenerator(
            data_paths=paths,
            num_validation_samples=1,
            batch_size=32,
            epochs=50,
            max_queue_length=8096,
            data_processing_function=dli.get_sensordata_and_flowfront,
            data_gather_function=dg.get_filelist_within_folder,
            num_workers=4,
            cache_path=cache_path,
        )
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception("Fatal Error:")
        exit()
    return generator


if __name__ == "__main__":
    logging_cfg.apply_logging_config(None)
    logger = logging.getLogger(__name__)
    gen = create_datagenerator_pressure_flowfront()
    for inputs, labels in gen:
        print(np.shape(inputs), np.shape(labels))
        a = inputs.numpy()
        c = labels.numpy()
        b = np.reshape(a, [-1, 38, 30])
        b = b[:, ::4, ::4]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("colorMap")
        for i in range(32):
            plt.imshow(b[i])
            plt.show()
            plt.imshow(c[i])
            plt.show()

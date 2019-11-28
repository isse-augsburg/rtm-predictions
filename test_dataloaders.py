from pathlib import Path

from Pipeline import (
    torch_datagenerator as td,
    erfh5_pipeline as pipeline,
    data_gather as dg,
    data_loader_dryspot,
    data_loaders_IMG,
)
from Utils import logging_cfg

import time

if __name__ == "__main__":
    logging_cfg.apply_logging_config(None)
    _data_root = Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes")
    data_paths = [
        # _data_root / "2019-07-23_15-38-08_5000p",
        _data_root / "2019-07-24_16-32-40_5000p",
        # _data_root / "2019-07-29_10-45-18_5000p",
        # _data_root / "2019-08-23_15-10-02_5000p",
        # _data_root / "2019-08-24_11-51-48_5000p",
        # _data_root / "2019-08-25_09-16-40_5000p",
        # _data_root / "2019-08-26_16-59-08_6000p",
        # _data_root / '2019-09-06_17-03-51_10000p',
        _data_root / '2019-11-08_15-40-44_5000p'
    ]
    batch_size = 128
    num_workers = 16
    load_data = data_loader_dryspot.get_flowfront_bool_dryspot_143x111
    num_runs = 1
    epochs = 2
    cache_path = Path("/cfs/share/cache/output_johannes/cache")

    total_time = 0
    total_samples = 0
    for i in range(num_runs):
        start = time.time()
        generator = td.LoopingDataGenerator(data_paths, dg.get_filelist_within_folder, load_data, batch_size=batch_size,
                                            epochs=epochs, num_workers=num_workers,
                                            cache_path=cache_path, cache_mode=td.CachingMode.FileList)
        for i in generator:
            total_samples += batch_size

        end = time.time()
        total_time += end - start
        print(f"Iteration done. Stats: {total_samples} samples; {total_time}s elapsed")
    print(f"Average time for new loader: {total_time/num_runs}s; average samples: {total_samples/num_runs}")

    total_time = 0
    total_samples = 0
    for i in range(num_runs):
        start = time.time()
        generator = pipeline.ERFH5DataGenerator(
            data_paths=data_paths,
            num_validation_samples=0,
            num_test_samples=0,
            batch_size=batch_size,
            epochs=1,
            max_queue_length=1000000,
            data_processing_function=load_data,
            data_gather_function=dg.get_filelist_within_folder,
            num_workers=num_workers,
            cache_path=cache_path,
            cache_mode=pipeline.CachingMode.FileList,
            save_path=None,
            load_datasets_path=None,
            test_mode=False,
        )

        for _ in generator:
            total_samples += batch_size

        end = time.time()
        total_time += end-start
        print(f"Iteration done. Stats: {total_samples} samples; {total_time}s elapsed")
    print(f"Average time for old loader: {total_time/num_runs}s; average samples: {total_samples/num_runs}")

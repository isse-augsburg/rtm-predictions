from pathlib import Path


def get_data_paths(_data_root: Path):
    data_paths = [
        # _data_root / "2019-07-23_15-38-08_5000p",     # Folder to play with, CAUTION, has invalid data
        #                                               # Dryspot Data  # Has blacklist # FVC       # Dry Spot Prob
        _data_root / "2019-07-24_16-32-40_5000p",       # X             # X             # .2 - .8   # High
        # _data_root / "2019-07-29_10-45-18_5000p",     # -             #               #
        # _data_root / "2019-08-23_15-10-02_5000p",     # -             #               #
        # _data_root / "2019-08-24_11-51-48_5000p",     # -             #               #
        # _data_root / "2019-08-25_09-16-40_5000p",     # -             #               #
        # _data_root / "2019-08-26_16-59-08_6000p",     # -             #               #
        # _data_root / '2019-09-06_17-03-51_10000p',    # in the making # X             # .3 - .8   # Very High
        _data_root / '2019-11-08_15-40-44_5000p'        # X             # X             # .3 - .5   # Low
    ]
    return data_paths

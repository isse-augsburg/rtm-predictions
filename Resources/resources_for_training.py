import getpass
import socket
from pathlib import Path

_home = Path('/cfs/home')
if socket.gethostname() == "swt-dgx1":
    cache_path = None
    save_path = Path(f"/cfs/share/cache/output_{getpass.getuser()}")

elif socket.gethostname() == "swtse130":
    _home = Path('X:/')
    cache_path = None
    save_path = Path(r"C:\Users\stiebesi\CACHE\train_out")

elif getpass.getuser() == 'lodes':
    save_path = Path('/cfs/share/cache/output_lukas/Local')
    cache_path = Path('/cfs/share/cache')

_stiebesi_home = _home / 's/t/stiebesi'
data_root = _stiebesi_home / 'data/RTM/Leoben/output/with_shapes'
datasets_dryspots = _stiebesi_home / 'data/RTM/Leoben/reference_datasets/dryspot_detection'
checkpoint_1140_sensors_deconv = _stiebesi_home / 'results_leoben/4_three_week_run/2019-09-25_16-42-53/checkpoint.pth'


def get_all_data_paths():
    data_paths = [
        data_root / "2019-07-23_15-38-08_5000p",     # Folder to play with, CAUTION, has invalid data
                                                     # Dryspot Data  # Has blacklist # FVC       # Dry Spot Prob
        data_root / "2019-07-24_16-32-40_5000p",     # X             # X             # .2 - .8   # High
        data_root / "2019-07-29_10-45-18_5000p",     # -             #               #
        data_root / "2019-08-23_15-10-02_5000p",     # -             #               #
        data_root / "2019-08-24_11-51-48_5000p",     # -             #               #
        data_root / "2019-08-25_09-16-40_5000p",     # -             #               #
        data_root / "2019-08-26_16-59-08_6000p",     # -             #               #
        data_root / '2019-09-06_17-03-51_10000p',    # X             # X             # .3 - .8   # Very High
        data_root / '2019-11-08_15-40-44_5000p'      # X             # X             # .3 - .5   # Low
    ]
    return data_paths


def get_data_paths():
    data_paths = [
        #                                               # Dryspot Data  # Has blacklist # FVC       # Dry Spot Prob
        data_root / "2019-07-24_16-32-40_5000p",       # X             # X             # .2 - .8   # High
        data_root / '2019-11-08_15-40-44_5000p'        # X             # X             # .3 - .5   # Low
    ]
    return data_paths
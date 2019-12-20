import getpass
import socket
from pathlib import Path

_home = Path('/cfs/home')
if "swt-dgx" in socket.gethostname():
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
_res_leoben = _stiebesi_home / 'results_leoben'
data_root = _stiebesi_home / 'data/RTM/Leoben/output/with_shapes'
datasets_dryspots = _stiebesi_home / 'data/RTM/Leoben/reference_datasets/dryspot_detection'
checkpoint_1140_sensors_deconv = _res_leoben / '4_three_week_run/2019-09-25_16-42-53/checkpoint.pth'
checkpoint_1140_sensors_deconv_eff = _res_leoben / '2019-12-12_20-27-20_eff_net_cleaned_data/checkpoint.pth'
chkp_1140_transfered_dry_spot = _res_leoben / '2019-12-13_17-15-26_transfer_eff_net_dryspot/checkpoint.pth'


def get_all_data_paths():
    data_paths = [
        # data_root / "2019-07-23_15-38-08_5000p",     # Folder to play with, CAUTION, has invalid data
        # 1.379.230                                    # Dryspot  # Has blacklist # FVC       # Dry Spot Prob # Useless
        data_root / "2019-07-24_16-32-40_5000p",  # X        # X             # .2 - .8   # High          # X
        # 1.391.145
        data_root / "2019-07-29_10-45-18_5000p",  # started        #               #
        # 496.161
        data_root / "2019-08-23_15-10-02_5000p",  # started        #               #
        # 497.478
        data_root / "2019-08-24_11-51-48_5000p",  # started        #               #
        # 499.598
        data_root / "2019-08-25_09-16-40_5000p",  # started        #               #
        # 1.729.106
        data_root / "2019-08-26_16-59-08_6000p",  # started        #               #
        # 1.606.405
        data_root / '2019-09-06_17-03-51_10000p',  # X        # X             # .3 - .8   # Very High
        # 816.891
        data_root / '2019-11-08_15-40-44_5000p',  # X        # X             # .3 - .5   # Low           # X
        # 1.406.003
        data_root / '2019-11-29_16-56-17_10000p'  # started        # -             #
        # Overall count: 9822017
    ]
    return data_paths


def get_data_paths():
    data_paths = [
        #                                               # Dryspot Data  # Has blacklist # FVC       # Dry Spot Prob
        data_root / "2019-07-24_16-32-40_5000p",  # X             # X             # .2 - .8   # High
        data_root / '2019-11-08_15-40-44_5000p'  # X             # X             # .3 - .5   # Low
    ]
    return data_paths


def get_example_erfh5():
    return data_root / "2019-07-24_16-32-40_5000p" / '0' / '2019-07-24_16-32-40_0_RESULT.erfh5'

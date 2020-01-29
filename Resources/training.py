import getpass
import socket
from pathlib import Path

_home = Path('/cfs/home')
_share_path = Path("/cfs/share")
if "swt-dgx" in socket.gethostname():
    cache_path = None
    save_path = Path(f"/cfs/share/cache/output_{getpass.getuser()}")

elif socket.gethostname() == "swtse130":
    _home = Path('X:/')
    cache_path = None
    save_path = Path(r"C:\Users\stiebesi\CACHE\train_out")
    _share_path = Path(r"Y:/")

else:
    save_path = Path(f'/cfs/share/cache/output_{getpass.getuser()}/Local')
    # cache_path = Path('/cfs/share/cache')
    cache_path = None

_stiebesi_home = _home / 's/t/stiebesi'
_output_stiebesi = _share_path / Path('cache/output_stiebesi')
_results = _stiebesi_home / 'results'
_ijcai = _results / 'IJCAI_PRICAI_20_FlowFrontNet'
_ij_deconv_conv = _ijcai / "S1140_to_DS_deconv_conv"

data_root = _stiebesi_home / 'data/RTM/Leoben/output/with_shapes'

datasets_dryspots = _stiebesi_home / 'data/RTM/Leoben/reference_datasets/dryspot_detection'

chkp_S1140_to_ff_deconv = _results / '4_three_week_run/2019-09-25_16-42-53/checkpoint.pth'
chkp_S1140_to_ff_eff = _results / '2019-12-12_20-27-20_eff_net_cleaned_data/checkpoint.pth'
chkp_1140_transferred_dry_spot = _ij_deconv_conv / '2019-12-13_17-15-26_transfer_eff_net_dryspot/checkpoint.pth'
chkp_99_5_acc_retrained = _results / '2019-12-20_16-35-22_99.5_acc/checkpoint.pth'
chkp_S1140_to_ff_correct_data = _ij_deconv_conv / '2020-01-18_12-34-13_S1140_to_ff_bs1024_best_loss' \
                                                  '/checkpoint.pth'
chkp_S1140_to_ds_frozen = _ij_deconv_conv / "2020-01-21_09-32-50_S1140_to_DS_frozen_bad_chkp/checkpoint.pth"
chkp_S1140_to_ds_frozen_deeper_convnet = _ij_deconv_conv / "2020-01-22_16-44-26_S1140_to_DS_frozen_deeper_convnet" \
                                                           '/checkpoint.pth'
chkp_S1140_to_ds_frozen_deeper_convnet2 = _ij_deconv_conv / "2020-01-22_16-44-26_S1140_to_DS_frozen_deeper_convnet" \
                                                            '/checkpoint_best_val_loss.pth'

chkp_20_sensors_to_ff = _output_stiebesi / '2020-01-10_15-09-30/checkpoint0_2055val.pth'


def get_all_data_paths():
    data_paths = [
        # data_root / "2019-07-23_15-38-08_5000p",     # Folder to play with, CAUTION, has invalid data
        # 1.379.230                               # Dryspot  # Has blacklist # FVC       # Dry Spot Prob # Useless
        data_root / "2019-07-24_16-32-40_5000p",  # X        # X             # .2 - .8   # High          # X
        # 1.391.145
        data_root / "2019-07-29_10-45-18_5000p",  # X        # X             # .2 - .8   # High          # X
        # 496.161
        data_root / "2019-08-23_15-10-02_5000p",  # started  #               #
        # 497.478
        data_root / "2019-08-24_11-51-48_5000p",  # started  #               #
        # 499.598
        data_root / "2019-08-25_09-16-40_5000p",  # started  #               #
        # 1.729.106
        data_root / "2019-08-26_16-59-08_6000p",  # started  #               #
        # 1.606.405
        data_root / '2019-09-06_17-03-51_10000p',  # X       # X             # .3 - .8   # Very High
        # 816.891
        data_root / '2019-11-08_15-40-44_5000p',  # X        # X             # .3 - .5   # Low           # X
        # 1.406.003
        data_root / '2019-11-29_16-56-17_10000p'  # X        # X             # .3 - .5   # Low           # X
        # Overall count: 9822017
    ]
    return data_paths


def get_data_paths():
    data_paths = [
        #                                         # Dryspot Data  # Has blacklist # FVC       # Dry Spot Prob
        data_root / "2019-07-24_16-32-40_5000p",  # X             # X             # .2 - .8   # High
        data_root / '2019-11-08_15-40-44_5000p'  # X             # X             # .3 - .5   # Low
    ]
    return data_paths


def get_data_paths_debug():
    data_paths = [
        #                                       # Dryspot Data  # Has blacklist # FVC       # Dry Spot Prob
        data_root / "2019-07-24_16-32-40_10p",  # X             # X             # .2 - .8   # High
    ]
    return data_paths


def get_data_paths_new_test_set():
    data_paths = [
        data_root / "2019-07-29_10-45-18_5000p",  # X        # X             # .2 - .8   # High          # X
        data_root / '2019-11-29_16-56-17_10000p'  # X        # X             # .3 - .5   # Low           # X
    ]
    return data_paths


def get_example_erfh5():
    return data_root / "2019-07-24_16-32-40_5000p" / '0' / '2019-07-24_16-32-40_0_RESULT.erfh5'

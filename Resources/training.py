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
output_stiebesi = _share_path / Path('cache/output_stiebesi')
_results = _stiebesi_home / 'results'
_ijcai = _results / 'IJCAI_PRICAI_20_FlowFrontNet'
_ij_S1140_deconv_conv = _ijcai / "S1140_to_DS_deconv_conv"
_ij_S20_deconv_conv = _ijcai / "S20_to_DS_deconv_conv"
_ij_S80_deconv_conv = _ijcai / "S80_to_DS_deconv_conv"
_ij_densenet_baselines = _ijcai / "densenet_baseline"
plots_output = _ijcai / "plots"

datasets_dryspots = _share_path / 'data/RTM/Leoben/reference_datasets/dryspot_detection'
datasets_dryspots_torch = _share_path / 'data/RTM/Leoben/reference_datasets_torch'
chkp = "checkpoint.pth"

# chkp_S1140_to_ff_deconv = _results / "4_three_week_run/2019-09-25_16-42-53" / chkp
chkp_S1140_to_ff_eff = _results / "2019-12-12_20-27-20_eff_net_cleaned_data" / chkp
# chkp_S1140_transferred_dry_spot = _ij_deconv_conv / '2019-12-13_17-15-26_transfer_eff_net_dryspot/checkpoint.pth'
# Mixed up data sets in different training phases:
# chkp_99_5_acc_retrained = _results / '2019-12-20_16-35-22_99.5_acc/checkpoint.pth'
chkp_S1140_to_ff_correct_data = _ij_S1140_deconv_conv / \
    '2020-01-18_12-34-13_S1140_to_ff_bs1024_best_loss' / chkp
# chkp_S1140_to_ff_retrain_mixed_press = _ij_S1140_deconv_conv / \
#     "0_new_split_mixed_ground_pressure/" \
#     "2020-02-03_17-19-34_S1140_to_ff_retrain" / chkp
zero_basepr = _ij_S1140_deconv_conv / "1_new_split_0_ground_pressure"
dir_S1140_to_ds = zero_basepr / "2020-02-12_11-23-35_S1140_to_ds_base_0_frozen"
chkp_S1140_to_ff_0_basepr = zero_basepr / "2020-02-07_13-55-43_S1140_to_ff_base_0" / chkp
chkp_S1140_to_ds_0_basepr_frozen = zero_basepr / "2020-02-12_11-23-35_S1140_to_ds_base_0_frozen" / chkp
chkp_S1140_to_ds_frozen = _ij_S1140_deconv_conv / "2020-01-21_09-32-50_S1140_to_DS_frozen_bad_chkp" / chkp
chkp_S1140_to_ds_frozen_deeper_convnet = _ij_S1140_deconv_conv / \
    "2020-01-22_16-44-26_S1140_to_DS_frozen_deeper_convnet" / chkp
chkp_S1140_to_ds_frozen_deeper_convnet2 = _ij_S1140_deconv_conv / \
    "2020-01-22_16-44-26_S1140_to_DS_frozen_deeper_convnet" / \
    "checkpoint_best_val_loss.pth"
chkp_S1140_densenet_baseline = _ij_densenet_baselines / "2020-01-17_09-41-19_S1140_to_DS_densenet_32768bs" / chkp
chkp_S1140_densenet_baseline_full_trainingset = \
    _ij_densenet_baselines / "2020-03-26_16-51-56_S1140_to_DS_densenet_32768bs_full_training" / chkp

chkp_S20_densenet_baseline = _ij_densenet_baselines / "2020-01-17_12-03-17_S20_to_DS_densenet_wide_32768bs" / chkp
chkp_S20_densenet_baseline_full_trainingset = \
    _ij_densenet_baselines / "2020-03-26_16-39-42_S20_to_DS_densenet_32768bs_full_training_set" / chkp
chkp_S20_to_ff = output_stiebesi / "2020-01-10_15-09-30/checkpoint0_2055val.pth"
chkp_S20_to_ff_basepr_0 = _ij_S20_deconv_conv / "2020-02-14_16-05-28_S_to_FF" / chkp
chkp_S20_to_ds = _ij_S20_deconv_conv / "2020-02-18_08-34-18_S_to_DS" / chkp
chkp_S20_to_ds_retrain = _ij_S20_deconv_conv / "2020-03-27_12-21-34_retrain_s_to_ds" / chkp

chkp_S80_densenet_baseline = _ij_densenet_baselines / "2020-02-26_14-05-01_S80_to_DS_densenet_wide_32768bs" / chkp
chkp_S80_to_ff = _ij_S80_deconv_conv / "2020-02-26_17-12-48_deconv_first_try" / chkp
chkp_S80_to_ff2 = _ij_S80_deconv_conv / "2020-02-28_12-07-38_deconv_second_try" / chkp
chkp_S80_to_ds_no_thres = _ij_S80_deconv_conv / "2020-03-02_14-19-12_conv_second_try_no_threshold" / chkp
dir_S80_to_ds_thres = _ij_S80_deconv_conv / "2020-03-04_10-41-02_th_0.8_best"
chkp_S80_to_ds_thres = _ij_S80_deconv_conv / "2020-03-04_10-41-02_th_0.8_best" / chkp
chkp_S80_to_ds_thres_longer_train = \
    _ij_S80_deconv_conv / "2020-03-10_17-52-15_S80_to_DS_th_0.8_longer_train_no_lr_scheduler" / chkp
chkp_S80_to_ff_standardized = _ij_S80_deconv_conv / "standardized/2020-03-09_18-14-25_S80_to_ff_standardized" / chkp

data_root = _share_path / 'data/RTM/Leoben/sim_output'
nearest_nodes_to_sensors = _share_path / "data/RTM/Leoben/Mesh/nearest_nodes_to_sensors.p"
mean_std_20_flowfront_sensors = _share_path / "data/RTM/Leoben/aux_data/mean_std_20_flowfront_sensors.p"
mean_std_1140_pressure_sensors = _share_path / "data/RTM/Leoben/aux_data/mean_std_1140_pressure_sensors.p"


def get_all_data_paths():
    data_paths = [
        # data_root / "2019-07-23_15-38-08_5000p",     # Folder to play with, CAUTION, has invalid data
        #                                          # DS  | BL | FVC       | Drysp Prob | Usel.  | Press | Fr. Count
        data_root / "2019-07-24_16-32-40_5000p",   # X   | X  | .2 - .8   | High      | X       | 0     | 1.379.230
        data_root / "2019-07-29_10-45-18_5000p",   # X   | X  | .2 - .8   | High      | X       | 0     | 1.391.145
        data_root / "2019-08-23_15-10-02_5000p",   # X   | X  | .1 - .8   | High      | X       | 0     | 496.161
        data_root / "2019-08-24_11-51-48_5000p",   # X   | X  | .1 - .8   | High      | X       | 0     | 497.478
        data_root / "2019-08-25_09-16-40_5000p",   # X   | X  | .1 - .8   | High      | X       | 0     | 499.598
        data_root / "2019-08-26_16-59-08_6000p",   # X   | X  | .3 - .8   | V. High   | X       | 0     | 1.729.106
        data_root / '2019-09-06_17-03-51_10000p',  # X   | X  | .3 - .8   | V. High   | X       | 0     | 1.606.405

        data_root / '2019-11-08_15-40-44_5000p',   # X   | X  | .3 - .5   | Low       | X       | 200k  | 816.891
        data_root / '2019-11-29_16-56-17_10000p'   # X   | X  | .3 - .5   | Low       | X       | 200k  | 1.406.003

        #                                                                                 Overall count:  9.822.017
    ]
    return data_paths


def get_data_paths():
    data_paths = [
        #                                         # Dryspot Data  # Has blacklist # FVC       # Dry Spot Prob
        data_root / "2019-07-24_16-32-40_5000p",  # X             # X             # .2 - .8   # High
        data_root / '2019-11-08_15-40-44_5000p'  # X             # X             # .3 - .5   # Low
    ]
    return data_paths


# 5000 * 3 runs + 8500 runs = 23500 runs;
# 1.379.230 + 1.391.145 + 816.891 + 1.406.003 = 4993269 ~ 5 M Samples - invalid samples from blacklist and useless
# Exactly: 3171683 Samples
# Test Train Split:
# Validation set 2.8 %:   131072
# Test set 23 %:         1048576

def get_more_data_paths():
    data_paths = [
        #                                         # DS  # BL # FVC       # Drysp Prob # Useless # Base Pressure
        data_root / "2019-07-24_16-32-40_5000p",  # X   # X  # .2 - .8   # High       # X       # 0
        data_root / '2019-11-08_15-40-44_5000p',  # X   # X  # .3 - .5   # Low        # X       # 200k
        data_root / "2019-07-29_10-45-18_5000p",  # X   # X  # .2 - .8   # High       # X       # 0
        data_root / '2019-11-29_16-56-17_10000p'  # X   # X  # .3 - .5   # Low        # X       # 200k
    ]
    return data_paths


def get_data_paths_base_200k():
    data_paths = [
        #                                         # DS  # BL # FVC       # Drysp Prob # Useless # Base Pressure
        data_root / '2019-11-08_15-40-44_5000p',  # X   # X  # .3 - .5   # Low        # X       # 200k
        data_root / '2019-11-29_16-56-17_10000p'  # X   # X  # .3 - .5   # Low        # X       # 200k
    ]
    return data_paths


# Exactly 5067352 Samples
# Test Train Split:
# Validation set 2.5 %:   131072
# Test set 20.6 %:         1048576

def get_data_paths_base_0():
    data_paths = [
        # 5067352                                  # DS  | BL | FVC       | Drysp Prob | Usel.  | Press | Fr. Count
        data_root / "2019-07-24_16-32-40_5000p",   # X   | X  | .2 - .8   | High      | X       | 0     | 1.379.230
        data_root / "2019-07-29_10-45-18_5000p",   # X   | X  | .2 - .8   | High      | X       | 0     | 1.391.145
        data_root / "2019-08-23_15-10-02_5000p",   # X   | X  | .1 - .8   | High      | X       | 0     | 496.161
        data_root / "2019-08-24_11-51-48_5000p",   # X   | X  | .1 - .8   | High      | X       | 0     | 497.478
        data_root / "2019-08-25_09-16-40_5000p",   # X   | X  | .1 - .8   | High      | X       | 0     | 499.598
        data_root / "2019-08-26_16-59-08_6000p",   # X   | X  | .3 - .8   | V. High   | X       | 0     | 1.729.106
        data_root / '2019-09-06_17-03-51_10000p',  # X   | X  | .3 - .8   | V. High   | X       | 0     | 1.606.405
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

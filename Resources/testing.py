import os
from pathlib import Path


def running_in_docker():
    path = '/proc/self/cgroup'
    return os.path.exists('/.dockerenv') or os.path.isfile(path) and any('docker' in line for line in open(path))


test_src_dir = Path(r'/cfs/home/s/t/stiebesi/code/tests/test_data')
torch_datasets = test_src_dir / "TestSaveDatasetsTorch" / "unix"
if running_in_docker():
    test_out_dir = Path('/cache')
# else:
# test_out_dir = Path(f'/cfs/share/cache/output_johanmay/tests')

if os.name == "nt":
    test_out_dir = Path(r'C:\Users\stiebesi\CACHE\test_output')
    test_src_dir = Path(r'X:\s\t\stiebesi\code\tests\test_data')
    torch_datasets = test_src_dir / "TestSaveDatasetsTorch" / "win"

test_training_src_dir = test_src_dir
test_pipeline_dir = test_src_dir

test_eval_dir = test_src_dir / '2019-12-12_20-27-20_eff_net_cleaned_data'
test_checkpoint = test_src_dir / '2019-12-12_20-27-20_eff_net_cleaned_data' / "checkpoint.pth"

data_loader_img_file = test_src_dir / '2019-06-05_15-30-52_0_RESULT.erfh5'

test_eval_output_path = test_out_dir / 'eval'
test_save_dataset_path = test_out_dir / 'dataset'
test_training_out_dir = test_out_dir / 'training'
test_training_datasets_dir = test_out_dir / 'datasets'

test_split_path = test_src_dir / '2019-09-06_15-44-58_63_sensors'

test_caching_dir = test_out_dir / 'erfh5_pipeline' / 'caching'

test_useless_file = test_src_dir / "2019-07-23_15-38-08_5000p_0/2019-07-23_15-38-08_0_RESULT.erfh5"


if __name__ == "__main__":
    print(running_in_docker())

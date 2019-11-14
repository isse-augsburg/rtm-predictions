import getpass
import os
from pathlib import Path


def running_in_docker():
    path = '/proc/self/cgroup'
    return os.path.exists('/.dockerenv') or os.path.isfile(path) and any('docker' in line for line in open(path))


if getpass.getuser() == 'stiebesi':
    test_out_dir = Path(r'C:\Users\stiebesi\CACHE\test_output')
    test_src_dir = Path(r'X:\s\t\stiebesi\code\tests\test_data')
elif getpass.getuser() == 'lodes':
    test_out_dir = Path(r'/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/l/o/lodesluk/code/tests')
    test_src_dir = Path(r'/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/code/tests'
                        '/test_data')
elif getpass.getuser() == 'Lukas':
    test_out_dir = Path(r'X:\l\o\lodesluk\code\tests')
    test_src_dir = test_src_dir = Path(r'X:\s\t\stiebesi\code\tests\test_data')
elif running_in_docker():
    test_src_dir = Path('/cfs/home/s/t/stiebesi/code/tests/test_data')
    test_out_dir = Path('/cache')
else:
    raise Exception("User not defined. Please define username.")

test_training_src_dir = test_src_dir
test_pipeline_dir = test_src_dir

test_eval_dir = test_src_dir / '2019-09-02_19-40-56'
data_loader_img_file = test_src_dir / '2019-06-05_15-30-52_0_RESULT.erfh5'

test_eval_output_path = test_out_dir / 'eval'
test_training_out_dir = test_out_dir / 'training'

test_caching_dir = test_out_dir / 'erfh5_pipeline' / 'caching'

if __name__ == "__main__":
    print(running_in_docker())

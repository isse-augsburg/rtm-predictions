from pathlib import Path
import getpass

if getpass.getuser() == 'stiebesi':
    test_out_dir = Path(r'X:\s\t\stiebesi\code\tests')
    test_src_dir = Path(r'X:\s\t\stiebesi\code\tests')
elif getpass.getuser() == 'lodesluk':
    test_out_dir = "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/l/o/lodesluk/code/tests"
    test_src_dir = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/code/tests'
else:
    raise Exception("User not defined. Please define username.")

test_training_src_dir = test_src_dir / 'training_data_source'
test_eval_dir = test_src_dir / Path('evaluation/2019-09-02_19-40-56')
data_loader_img_file = test_src_dir / 'data_loader_IMG' / '2019-06-05_15-30-52_0_RESULT.erfh5'

test_training_out_dir = test_out_dir / 'training'

test_pipeline_dir = test_out_dir / Path('erfh5_pipeline/source')
test_caching_dir = test_out_dir / Path('erfh5_pipeline/caching')


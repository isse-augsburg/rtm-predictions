import getpass
import os
from pathlib import Path

if getpass.getuser() == 'lodes':
    train_out_dir = Path('/home/lodes/Train_Out')
    train_data_dir = Path(
            '/run/user/1001/gvfs/smb-share:server=137.250.170.56,'
            'share=home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes')
    cache_dir = Path(
            '/run/user/1001/gvfs/smb-share:server=137.250.170.56,'
            'share=share/cache')
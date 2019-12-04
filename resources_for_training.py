import getpass
from pathlib import Path
import socket


if socket.gethostname() == "swt-dgx1":
    print("On DGX.")
    cache_path = None
    data_root = Path(
        "/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes")  
    if getpass.getuser() == "lodesluk":
        save_path = Path("/cfs/share/cache/output_lukas")
    else:
        save_path = Path("/cfs/share/cache/output")

elif getpass.getuser() == 'lodes':
    save_path = Path('/cfs/share/cache/output_lukas/Local')
    data_root = Path(
        '/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes')
    cache_path = Path('/cfs/share/cache')

import logging
from pathlib import Path


def apply_logging_config(save_path):
    logfile = None
    if save_path is not None:
        logfile = Path(save_path) / "output.log"
    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if save_path is not None:
        # Add a default stream handler to the root logger - this will propagate to all other loggers
        sh = logging.StreamHandler()
        sh.setLevel(logging.ERROR)
        logging.getLogger().addHandler(sh)

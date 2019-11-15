import logging
from pathlib import Path


def apply_logging_config(save_path, eval=False):
    logfile = None
    if save_path is not None:
        filename = "output.log" if not eval else "test_output.log"
        logfile = Path(save_path) / filename

    # Clear existing config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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

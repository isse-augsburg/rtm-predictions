import argparse
import logging
import sys
from enum import Enum

import numpy as np
import torch


class CheckpointingStrategy(Enum):
    """
    Enum for specifying which checkpoints are stored during training.
    Best: Only the checkpoint with model's best performance is stored.
    All: All checkpoints are stored.
    """
    Best = 1
    All = 2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transform_to_tensor_and_cache(i, separate_set_list, num=0, s_path=None):
    _data = torch.FloatTensor(i[0])
    # The following if else is necessary to have 0, 1 Binary Labels in Tensors
    # since FloatTensor(0) = FloatTensor([])
    if type(i[1]) is np.ndarray and len(i[1]) > 1:
        _label = torch.FloatTensor(i[1])
    else:
        if i[1] == 0:
            _label = torch.FloatTensor([0.])
        elif i[1] == 1:
            _label = torch.FloatTensor([1.])

    separate_set_list.append((_data, _label))
    if s_path is not None:
        s_path.mkdir(parents=True, exist_ok=True)
        torch.save(_data, s_path.joinpath(str(num) + "-data" + ".pt"))
        torch.save(_label, s_path.joinpath(str(num) + "-label" + ".pt"))


def apply_blacklists(_data_source_paths):
    _cleaned = []
    for _p in _data_source_paths:
        blacklist_f = _p / "blacklist.txt"
        if blacklist_f.exists():
            with open(blacklist_f) as f:
                lines = f.readlines()
                cleaned = [x.split(" ")[0] for x in lines]
                runs_num_only = set([int(x.split("/")[1]) for x in cleaned])
                for subdir in _p.iterdir():
                    if not subdir.is_dir():
                        continue
                    if int(subdir.stem) in runs_num_only:
                        continue
                    _cleaned.append(subdir)
        else:
            for subdir in _p.iterdir():
                if not subdir.is_dir():
                    continue
                _cleaned.append(subdir)

    return _cleaned


def read_cmd_params():
    parser = argparse.ArgumentParser(description="Run training or test. Rerun experiments from paper with data from"
                                                 "https://figshare.com/s/6d8ebc90e0e820b7f08f")
    parser.add_argument("--eval", type=str, default=None, help="Run a test. Full directory to output "
                                                               "trained model (to test).")
    parser.add_argument("--demo", type=str, default=None, help="Run experiments from FlowFrontNet paper. "
                                                               "Add the full directory path to dataset")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Full directory path to a checkpoint")

    args = parser.parse_args()
    if args.eval is None:
        args.run_eval = False
    else:
        args.run_eval = True
    eval_path = args.eval
    checkpoint_path = args.checkpoint_path

    if args.run_eval and (eval_path is None or checkpoint_path is None):
        logger = logging.getLogger(__name__)
        logger.error(
            "No eval_path or checkpoint_path given. You should specify the --eval_path / --checkpoint_path argument if"
            "you would like to run a test."
        )
        logger.error(parser.format_help())
        logging.shutdown()
        sys.exit()

    return args

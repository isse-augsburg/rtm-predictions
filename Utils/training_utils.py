import argparse
import logging
import sys

import numpy as np
import torch


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
        blacklist_f = _p / 'blacklist.txt'
        if blacklist_f.exists():
            with open(blacklist_f) as f:
                lines = f.readlines()
                cleaned = [x.split(' ')[0] for x in lines]
                runs_num_only = set([int(x.split('/')[1]) for x in cleaned])
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
    parser = argparse.ArgumentParser(description='Run training or test.')
    parser.add_argument('--eval', action='store_true', help='Run a test.')
    parser.add_argument('--eval_path', type=str, default=None, help='Full directory path of trained model (to test).')
    args = parser.parse_args()
    run_eval = args.eval
    eval_path = args.eval_path

    if run_eval and eval_path is None:
        logger = logging.getLogger(__name__)
        logger.error("No eval_path given. You should specify the --eval_path argument if you would like to run a test.")
        logger.error(parser.format_help())
        logging.shutdown()
        sys.exit()

    return args
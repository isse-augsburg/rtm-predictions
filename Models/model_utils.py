import logging
from collections import OrderedDict
from pathlib import Path

import torch


def load_model_layers_from_path(path: Path, layer_names: set):
    logger = logging.getLogger(__name__)
    logger.info(f'Loading model from {path}')
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')

    new_model_state_dict = OrderedDict()
    model_state_dict = checkpoint["model_state_dict"]
    for k, v in model_state_dict.items():
        splitted = k.split('.')
        name = splitted[1]  # remove `module.`
        if name in layer_names:
            print(f'{name}.{splitted[2]}')
            new_model_state_dict[f'{name}.{splitted[2]}'] = v
        else:
            continue
    return new_model_state_dict

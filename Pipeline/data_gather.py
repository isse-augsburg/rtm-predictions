from os import walk
from pathlib import Path


def get_filelist_within_folder(root_directory):
    """
    Args: 
        root_directory (string): Root directory from which all paths should be
        collected.

    Returns: 
        List of all .erfh5 files in the root_directory 
    """

    dataset_filenames = []

    for (dirpath, _, filenames) in walk(root_directory):
        if filenames:
            filenames = [Path(dirpath) / f for f in filenames if
                         f.endswith('.erfh5')]
            dataset_filenames.extend(filenames)

    return dataset_filenames


def get_filelist_within_folder_blacklisted(root_directory: str):
    """
    Args:
        root_directory (string): Root directory from which all paths should be
        collected.

    Applies a blacklist, located in the root directory to the paths and thus does not return them.

    Returns:
        List of all .erfh5 files in the root_directory
    """
    blacklist = set()
    blacklist_f = Path(root_directory) / 'blacklist.txt'
    if blacklist_f.exists():
        with blacklist_f.open('r') as f:
            lines = f.readlines()
            cleaned = [x.split(' ')[0] for x in lines]
            blacklist = set([int(x.split('/')[1]) for x in cleaned])
    dataset_filenames = []

    for (dirpath, _, filenames) in walk(root_directory):
        if Path(dirpath).stem.isdigit():
            irun = int(Path(dirpath).stem)
            if irun in blacklist:
                continue
        if filenames:
            filenames = [Path(dirpath) / f for f in filenames if f.endswith('.erfh5')]
            dataset_filenames.extend(filenames)

    return dataset_filenames


def get_folders_within_folder(root_directory):
    folders = list()

    for (dirpath, dirnames, _) in walk(root_directory):
        dirs = [dirpath + '/' + f for f in dirnames]
        folders.extend(dirs)
        break
    return folders

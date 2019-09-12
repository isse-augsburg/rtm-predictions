from os import walk
from pathlib import Path


def get_filelist_within_folder(root_directory):
    """
    Args: 
        root_directory (string): Root directory from which all paths should be collected. 

    Returns: 
        List of all .erfh5 files in the root_directory 
    """

    dataset_filenames = []

    for (dirpath, _, filenames) in walk(root_directory):
        if filenames:
            filenames = [Path(dirpath) / f for f in filenames if f.endswith('.erfh5')]
            dataset_filenames.extend(filenames)

    return dataset_filenames

    # l = []
    # for dir in root_directories:
    #    l.extend(list(Path(dir).glob('**/*.erfh5')))
    # return l


def get_folders_within_folder(root_directory):
    folders = list()

    for (dirpath, dirnames, _) in walk(root_directory):
        l = [dirpath + '/' + f for f in dirnames]
        folders.extend(l)
        break
    return folders

from os import walk
from pathlib import Path


def get_filelist_within_folder(root_directories):
    l = []
    for dir in root_directories:
        l.extend(list(Path(dir).glob('**/*.erfh5')))
    return l


def get_folders_within_folder(root_directory):
    folders = list()

    for dirs in root_directory:

        for (dirpath, dirnames, _) in walk(root_directory):
            l = [dirpath + '/' + f for f in dirnames]
            folders.extend(l)
            break
    return folders

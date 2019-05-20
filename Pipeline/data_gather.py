from os import walk
from pathlib import Path


# def get_filelist_within_folder(root_directories):
#     dataset_filenames = []
#     for dirs in root_directories:
#
#         for (dirpath, _, filenames) in walk(dirs):
#             if filenames:
#                 filenames = [dirpath + '/' + f for f in filenames if f.endswith('.erfh5')]
#                 dataset_filenames.extend(filenames)
#
#     return dataset_filenames


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

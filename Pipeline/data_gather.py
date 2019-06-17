from os import walk


def get_filelist_within_folder(root_directory):
    """
    Args: 
        root_directory (string): Root directory from which all paths should be collected. 

    Returns: 
        List of all .erfh5 files in the root_directory 
    """ 
    
    dataset_filenames = []
    for dirs in root_directory:

        for (dirpath, _, filenames) in walk(dirs):
            if filenames:
                filenames = [dirpath + '/' + f for f in filenames if f.endswith('.erfh5')]
                dataset_filenames.extend(filenames)

    return dataset_filenames


def get_folders_within_folder(root_directory):
    folders = list()

    for dirs in root_directory:

        for (dirpath, dirnames, _) in walk(root_directory):
            l = [dirpath + '/' + f for f in dirnames]
            folders.extend(l)
            break
    return folders
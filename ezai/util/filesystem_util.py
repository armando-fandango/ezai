import os
import shutil

def makedir(path):
    """
    Makes the directories in the tree of the path.
    :param path:
    :return: nothing
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def rm(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
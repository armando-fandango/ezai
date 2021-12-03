from pathlib import Path
# import shutil
from typing import Union


def mkdir(path: Union[str, Path], mode=0o777, parents=True, exist_ok=True):
    """
    Makes the directories in the tree of the path.
    :param exist_ok:
    :param parents:
    :param mode:
    :param path:
    :return: nothing
    """
    if isinstance(path, str):
        path = Path(str)
    path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
    return path

# def rm(path: Union[str, Path]):
#    shutil.rmtree(path)

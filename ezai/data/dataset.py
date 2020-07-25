import abc

from ezai.util import log_util

logger = log_util.get_logger()

# Design:
#   Two sets of datasets classes
#     - representing ML data with x,y
#       - to decide how to keep it
#       - persist on disk as NPZ
#     - representing specific data like RITIS or whatever
#       - table data: persist on disk as parquet, in memory as arrow tables or dask
#       - text data:
#       - image data:
#   On file system the data is stored like this:
#
#   datasets
#       <dataset-code> e.g. mnist or ritis
#           <subset> or <subset_subset...> e.g. pems or pems_d5 or  etc.
#           <>-raw for raw data downloaded or received in original archive
#               mostly downloaded only once
#           <>-prep-<> for raw data pre-processed in parquet
#               one set of raw data can be prepped in multiple prep folders
#           <>-num-<> for data in numeric format e.g. npz, ready for ML
#               one raw -> n prep -> m numeric formats
#           <>-exp-<experiment_id> for experiments e,g, pems_d5-n11_id30
#               on each numeric format we may run multiple experiments
#

class Dataset(metaclass = abc.ABCMeta):
    __slots__ = ()

class MLDataset(Dataset):
    """
    Numerical (Numpy array) in X,Y format
    Optimal Class to just store the needed parts for ML
    """
    __slots__=()

class RawDataset(Dataset):
    """
    This class represents original / raw data sets
    Use cases:
    - download the datasets
    - process it to get into specific format - text, image, temporal, sequence
    - filter, group, subset
    """
    __slots__=()



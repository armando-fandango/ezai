import os
import sys
import importlib
import gc
import time

from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
from six.moves import urllib

from archived_code.ezai.util.archive import archive_extract
from archived_code.ezai.util import filesystem_util
from archived_code.ezai.util import log_util

l = log_util.get_logger()

def m_info(m_list):
    if not isinstance(m_list,(list,set,tuple)):
        m_list = (m_list)
    for m in m_list:
        print('{} {}'.format(m.__name__,m.__version__))

def m_load(mname,mpath):
    m = None
    if mname not in sys.modules:
        if mpath not in sys.path:
            sys.path.append(mpath)
        m = importlib.import_module(mname)
        m_info([m])
    else:
        m = sys.modules[mname]
    return m

def m_reload(m):
    m = importlib.reload(m)
    print(m.__file__)

def m_reload_ipython(m):
    # quietly deep-reload
    from IPython.lib import deepreload
    stdout = sys.stdout
    sys.stdout = open('junk','w')
    deepreload.reload(m)
    sys.stdout = stdout

def dataset_download_kaggle(dataset_name,dest_dir,kaggle_user=None, kaggle_key=None,
                            download_again=False,
                            extract=False):
    if download_again or not os.path.exists(dest_dir):
        if kaggle_user is not None:
            os.environ['KAGGLE_USER'] = kaggle_user
        if kaggle_key is not None:
            os.environ['KAGGLE_KEY'] = kaggle_key
        import kaggle # because it tries to authenticate at import
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=dest_dir,
                                          unzip=extract)
    else:
        l.info('{} exists, thus nothing downloaded'.format(dest_dir))


def dataset_download(source_url, source_files, dest_dir, dest_files=None,
                     download_again=False,
                     download_continue=True,
                     extract=True):
    """Download the data from source url, unless it's already here.
    :param:
        source_url: url to download from if file doesn't exist.
        source_files: list of files to be downloaded
        dest_file: string, name of the file in the directory.
        dest_dir: string, path to working directory.
        force_download: overwrite even if its there
        force_extract: overwrite even if its there

    :return:
        Path to resulting file.
    """
    if download_again or download_continue or not os.path.exists(dest_dir):
        if dest_files is None:
            dest_files = source_files

        filesystem_util.makedir(dest_dir)

        # why are we dong this ?
        # downloaded_files = []

        for source_file, dest_file in zip(source_files, dest_files):
            orig = urllib.parse.urljoin(source_url, source_file)
            dest = os.path.join(dest_dir, dest_file)
            if download_again or not os.path.exists(dest):
                l.info('Downloading: {}'.format(orig))
                error_msg = 'URL fetch failure on {}: {}'

                try:
                    try:
                        downloaded_file, _ = urlretrieve(orig, dest,
                                                         reporthook=None)
                    except URLError as e:
                        raise Exception(
                            error_msg.format(orig, '\n'.join([e.errno, e.reason])))
                    except HTTPError as e:
                        raise Exception(error_msg.format(orig,
                                                         '\n'.join([e.errno, e.code,
                                                                    e.reason])))
                except (Exception, KeyboardInterrupt) as e:
                    filesystem_util.rm(dest)
                    raise
                statinfo = os.stat(dest)
                l.info('Downloaded : {} ({} bytes)'.format(dest,statinfo.st_size))
            else:
                l.info('Already exists: {}'.format(dest))
            # why are we doing this ?
            #downloaded_files.append(dest_file)
            if extract:
                archive_extract(ar_filename=dest, dest=dest_dir)
        # no need, calling program can add dest_dir + dest_files to fet what was downloaded
        #return downloaded_files
    else:
        l.info('{} exists, thus nothing downloaded'.format(dest_dir))

# to find if a package exists or not
#import importlib
#spam_spec = importlib.util.find_spec("spam")
#found = spam_spec is not None

def tvt_split(data, train_size=0.8, valid_size=0.1):
    if data is None:
        raise ValueError('No ml data found')

    if train_size > 1 or train_size <= 0:
        raise ValueError('train_size has to be between 0 and 1')

    if valid_size >= 1 or valid_size < 0:
        raise ValueError('valid_size has to be between 0 and 1')

    if train_size + valid_size > 1:
        raise ValueError('train_size + valid_size has to be between 0 and 1')

    N = data.shape[0]

    train_size = int(N * train_size)
    valid_size = int(N * valid_size)
    test_size = N - train_size - valid_size

    if (train_size > 0):
        train = data[0:train_size]
    else:
        train = None

    if (valid_size > 0):
        valid = data[train_size:train_size + valid_size]
    else:
        valid = None

    if (test_size > 0):
        test = data[train_size + valid_size:N]
    else:
        test = None
    return train, valid, test

class ExpTimer:
    def __init__(self):
        # in fractional seconds
        self.start_time = 0
        self.stop_time = 0

    def start(self, clean=True):
        if clean:
            gc.collect()
        gc.disable()
        self.start_time = time.process_time()

    def stop(self,clean=True):
        self.stop_time = time.process_time()
        gc.enable()
        if clean:
            gc.collect()
        return self.elapsedTime   # in seconds

    @property
    def elapsedTime(self):
        return self.stop_time - self.start_time

    @property
    def elapsedTimeInMin(self):
        return (self.stop_time - self.start_time) / 60

def hypothesis(p, h1='', alpha=0.05):
    if p > alpha:
        return False, 'not enough evidence for: {} (fail to reject H0)'.format(h1)
    else:
        return True, 'significant evidence for: {} (reject H0)'.format(h1)






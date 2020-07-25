import bz2
import gzip
import zipfile
import tarfile
import os

from ezai.util import filesystem_util


class Archive():
    def __init__(self, ar_filename):
        self.archive = ar_open(ar_filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.archive.close()

    def open(selfself, ar_filename):
        return ar_open(ar_filename)

    def contains(self, filename):
        return ar_contains_f(self.archive, filename)

    def open_file(self, filename):
        return ar_open(self.archive, filename)

    def contents(self):
        return ar_contents(self.archive)

    def extract_all(self, dest):
        return ar_extract_all(self.archive, dest)

    def extract_file(self, filename, dest):
        return ar_extract_file(self.archive, filename, dest)

def ar_open(ar_filename):
    """
    Function to open the archive
    :param ar_filename:
    :return:
    """
    if ar_filename.endswith('.bz2'):
        ar = tarfile.open(ar_filename, 'r:bz2')
    elif ar_filename.endswith('.gz'):
        ar = tarfile.open(ar_filename, 'r:gz')
    elif ar_filename.endswith('.xz'):
        ar = tarfile.open(ar_filename, 'r:xz')
    elif ar_filename.endswith('.zip'):
        ar = zipfile.ZipFile(ar_filename)
    else:
        ar = open(ar_filename)
    return ar
    """
    if filename.endswith('.bz2'):
        return gzip.GzipFile(filename)
    elif filename.endswith('.gz'):
        return gzip.GzipFile(filename)
    elif filename.endswith('.xz'):
        return bz2.BZ2File(filename)
    elif filename.endswith('.zip'):
        return zipfile.ZipFile(filename)
    else:
        return open(filename)
    """

def ar_contains_f(ar, f):
    """
    Function to check if the archive contains a file or not
    :param ar:
    :param f:
    :return:
    """
    if isinstance(ar, zipfile.ZipFile):
        success =  f in ar.namelist()
    elif isinstance(ar, (tarfile.TarFile)):
        success = f in ar.getnames()
    else:
        success = False
    return success


def ar_contents(ar):
    """
    Function to list the contents of an archive

    :param ar:
    :return:
    """
    if isinstance(ar, zipfile.ZipFile):
        namelist = ar.namelist()
    elif isinstance(ar, (tarfile.TarFile)):
        namelist =  ar.getnames()
    else:
        namelist = False
    return namelist

def ar_open_file(ar, filename):
    """
    Function to open a specific file in an archive

    :param ar:
    :param filename:
    :return:
    """
    if isinstance(ar, (zipfile.ZipFile, bz2.BZ2File, gzip.GzipFile)):
        ar_file =  ar.open(filename)
    elif isinstance(ar, (tarfile.TarFile)):
        ar_file =  ar.extractfile(filename)
    else:
        ar_file =  False
    return ar_file

def ar_extract_all(ar, dest):
    """
    Function to extract all the files in an archive

    :param ar: open archive handle
    :param dest:
    :return:
    """
    try:
        ar.extractall(dest)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
        # remove the destination folder/file
        filesystem_util.rm(dest)
        raise

def ar_extract_file(ar, filename, dest):
    """
    Function to extract one file in an archive

    :param ar: open archive handle
    :param dest:
    :return:
    """
    try:
        ar.extract(dest)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
        # remove the destination folder/file
        filesystem_util.rm(dest)
        raise

def archive_extract(ar_filename, dest='.'):

    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
    """
    ar = ar_open(ar_filename)
    ar_extract_all(ar,dest)



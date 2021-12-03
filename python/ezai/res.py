import time
import tracemalloc
import gc

class ResourceCounter:
    """
    This class allows to monitor the following resources for the purpose of logging:
    1. Time:
    2. NumpyMemory

    Usage:
    1. Create the rc object:
    rc = ResourceCounter(clean=True)

    clean=True by default, does the garbage collection when you start the counter

    2. start the counter:
    rc.start()

    3. Take snapshot of current usage anytime:
    time_since_start, current_memory, peak_memory = rc.snapshot()

    4. Stop and get final values:
    time_since_start, current_memory, peak_memory = rc.snapshot()

    Can be restarted anytime from zero by calling start.

    """
    def __init__(self, clean:bool=False):
        # in fractional seconds
        self.start_time = 0
        self.stop_time = 0
        self.clean = clean

    def _check_and_raise_start(self):
        if self.start_time <= 0:
            raise ValueError('ResourceCounter in invalid state. Did you forget to start the counter?')

    def _check_and_raise_stop(self):
        if self.stop_time < self.start_time:
            raise ValueError('ResourceCounter in invalid state. Did you forget to stop the counter?')

    def start(self):
        if self.clean:
            gc.collect()
        self.start_time = time.process_time()
        self.stop_time = 0
        tracemalloc.start()

    def snapshot(self):
        self._check_and_raise_start()
        snapshot_time = time.process_time()
        time_since_start = snapshot_time - self.start_time
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        return time_since_start, current_memory, peak_memory   # in seconds

    def stop(self):
        self._check_and_raise_start()
        time_since_start, current_memory, peak_memory = self.snapshot()
        self.stop_time = time_since_start + self.start_time
        tracemalloc.stop()
        if self.clean:
            gc.collect()
        return time_since_start, current_memory, peak_memory   # in seconds

    @property
    def elapsed_time(self):
        self._check_and_raise_start()
        self._check_and_raise_stop()

        return self.stop_time - self.start_time

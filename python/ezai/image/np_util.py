import numpy as np


def image_layout(x: np.ndarray, old: str, new: str) -> np.ndarray:
    new = [old.index(char) for char in new]
    return x.transpose(new)


def hwc_to_chw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        x_ = image_layout(x, 'hwc', 'chw')
    elif x.ndim == 4:
        x_ = image_layout(x, 'nhwc', 'nchw')
    else:
        raise ValueError(f"s has {x.ndim} dimensions, 3 or 4 dimensions needed")
    return x_


def chw_to_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        x_ = image_layout(x, 'chw', 'hwc')
    elif x.ndim == 4:
        x_ = image_layout(x, 'nchw', 'nhwc')
    else:
        raise ValueError(f"s has {x.ndim} dimensions, 3 or 4 dimensions needed")
    return x_


def s_hwc_to_chw(s):
    # TODO: HWC to CHW conversion optimized here
    # because pytorch can only deal with images in CHW format
    # we are making the optimization here to convert from HWC to CHW format.
    if isinstance(s, np.ndarray) and (s.ndim > 2):
        s = hwc_to_chw(s)
    elif isinstance(s, list):  # state is a list of states
        for i in range(len(s)):
            if isinstance(s[i], np.ndarray) and (s[i].ndim > 2):
                s[i] = hwc_to_chw(s[i])
    return s

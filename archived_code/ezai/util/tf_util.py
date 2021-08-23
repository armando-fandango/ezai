import tensorflow as tf

from archived_code.ezai.util import log_util
l = log_util.get_logger()

def image_layout(x, old, new):
    new = [old.index(char) for char in new]
    return tf.transpose(x,new)

def onehot(y, n_classes):
    return tf.one_hot(y,n_classes)

def argmax(x):
    return tf.argmax(x,axis=1)

def tfds(x,y):
    return tf.data.Dataset.from_tensor_slices((x, y))

def gpu_test():
    result = True
    if tf.test.is_built_with_cuda():
        l.info('TensorFlow CUDA version is installed')
        if tf.test.gpu_device_name():
            l.info('TensorFlow Default GPU Device:{}'.format(tf.test.gpu_device_name()))
            l.info('# of GPU Devices:{}'.format(len(tf.config.list_physical_devices('GPU'))))
        else:
            l.info('TensorFlow could not detect any GPU')
            result = False
    else:
        l.info('TensorFlow CPU version is installed')
        result = False

    return result
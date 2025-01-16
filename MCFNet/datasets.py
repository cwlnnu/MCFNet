import os

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
from itertools import count

import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat


def _california(reduce=False):
    mat = loadmat("data/California/UiT_HCD_California_2017.mat")

    t1 = tf.convert_to_tensor(mat["t1_L8_clipped"], dtype=tf.float32)
    t2 = tf.convert_to_tensor(mat["logt2_clipped"], dtype=tf.float32)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (4, 4)
        new_dims = list(map(lambda a, b: int(a // b), change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask


def _texas(clip=True):
    mat = loadmat("data/Texas/Cross-sensor-Bastrop-data.mat")
    t1 = np.array(mat["t1_L5"], dtype=np.single)
    t2 = np.array(mat["t2_ALI"], dtype=np.single)

    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI_1"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]

    return t1, t2, change_mask

def _shuguang(clip=True):
    mat = loadmat("data/Shuguang/Shuguang.mat")

    t1 = np.array(mat["shuguang_1"], dtype=np.single)
    t2 = np.array(mat["shuguang_2"], dtype=np.single)
    print('ss', t1.shape, t2.shape)

    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)

    change_mask = tf.convert_to_tensor(mat["cdata"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]

    return t1, t2, change_mask


def _clip(image):
    temp = np.reshape(image, (-1, image.shape[-1]))

    limits = tf.reduce_mean(temp, 0) + 3.0 * tf.math.reduce_std(temp, 0)
    for i, limit in enumerate(limits):
        channel = temp[:, i]
        channel = tf.clip_by_value(channel, 0, limit)
        ma, mi = tf.reduce_max(channel), tf.reduce_min(channel)
        channel = 2.0 * ((channel) / (ma)) - 1
        temp[:, i] = channel

    return tf.reshape(tf.convert_to_tensor(temp, dtype=tf.float32), image.shape)


def _training_data_generator(x, y, p, patch_size):
    patch_size = 50
    c_x, c_y = x.shape[2], y.shape[2]
    chs = c_x + c_y + 1
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    p_chs = slice(c_x + c_y, chs, 1)

    data = tf.concat([x, y, p], axis=-1)

    def gen():
        for _ in count():
            tmp = tf.image.random_crop(data, [patch_size, patch_size, chs])
            tmp = tf.image.rot90(tmp, np.random.randint(4))
            tmp = tf.image.random_flip_up_down(tmp)

            yield tmp[:, :, x_chs], tmp[:, :, y_chs], tmp[:, :, p_chs]

    dtypes = (tf.float32, tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
        tf.TensorShape([patch_size, patch_size, 1]),
    )

    return gen, dtypes, shapes


DATASETS = {
    "Texas": _texas,
    "California": _california,
    "Shuguang": _shuguang,
}
prepare_data = {
    "Texas": True,
    "California": True,
    "Shuguang": True,
}


def fetch(name, patch_size=100, **kwargs):
    x_im, y_im, target_cm = DATASETS[name](prepare_data[name])

    if not tf.config.list_physical_devices("GPU"):
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]

    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    x, y = dataset[0], dataset[1]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = x_im.shape[-1], y_im.shape[-1]

    return x, y, evaluation_data, (c_x, c_y)


if __name__ == "__main__":
    for DATASET in DATASETS:
        print(f"Loading {DATASET}")

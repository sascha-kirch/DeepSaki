"""Cutting operations performed on batched 3D grid-shaped data types like images."""
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf


def _random_boundingbox(height: int, width: int) -> Tuple[int, int, int, int]:
    """Generates a random bounding box that is smaller than the image.

    Args:
        height (int): Image height.
        width (int): Image width.

    Returns:
        x value top-left corner.
        y value top-left corner.
        x value bottom-right corner.
        y value bottom-right corner.
    """
    r = np.sqrt(1.0 - np.random.beta(1, 1))
    w = np.floor(width * r)
    h = np.floor(height * r)
    x = np.random.randint(width)
    y = np.random.randint(height)

    x1 = int(np.clip(x - w // 2, 0, width))
    y1 = int(np.clip(y - h // 2, 0, height))
    x2 = int(np.clip(x + w // 2, 0, width))
    y2 = int(np.clip(y + h // 2, 0, height))

    return x1, y1, x2, y2


def _get_mask(shape: Tuple[int, int, int, int]) -> tf.Tensor:
    """Generates a mask for given image dimensions (`batch_size`, `height`, `width`,`channel`).

    Args:
        shape (Tuple[int,int,int, int]): (`batch_size`, `height`, `width`,`channel`)

    Returns:
        Binary mask of shape (`batch_size`, `height`, `width`,`channel`).
    """
    mask = np.ones(shape=shape)
    for element in range(shape[0]):
        x1, y1, x2, y2 = _random_boundingbox(shape[1], shape[2])
        mask[element, x1:x2, y1:y2, :] = 0
    return tf.convert_to_tensor(mask, dtype=tf.float32)


def cut_mix(
    batch1: tf.Tensor,
    batch2: tf.Tensor,
    ignore_background: bool = False,
    invert_mask: bool = False,
    mask: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Performs the cutmix operation of two image batches.

    A random image patch from `batch2` is taken and inserted into `batch1.`

    Args:
        batch1 (tf.Tensor): Batch of grid-shaped data of shape (`batch`, `height`, `width`, `channel`).
        batch2 (tf.Tensor): Batch of grid-shaped data of shape (`batch`, `height`, `width`, `channel`).
        ignore_background (bool, optional): If true, pixels belonging to the backgroud are ignored. Only applicable for
            images where the background is represented as 0. Defaults to False.
        invert_mask (bool, optional): If true, the mask is inverted. 1->0 and 0->1. Defaults to False.
        mask (Optional[tf.Tensor], optional): Binary mask that requires same shape as `batch1` and `batch2`. If `None`
            mask is generated randomly. Defaults to None.

    Returns:
        ground_truth_mask: Actual mask that has been applied
        new_batch: Batch with applied cutmix opperation
    """
    batch1 = tf.cast(batch1, tf.float32)
    batch2 = tf.cast(batch2, tf.float32)

    if mask is None:  # generate mask
        mask = _get_mask(shape=batch1.shape)

    if ignore_background:  # check where in image are no background pixels (value = 1)
        batch1_mask = tf.cast(tf.where(batch1 > 0, 1, 0), tf.int32)
        batch2_mask = tf.cast(tf.where(batch2 > 0, 1, 0), tf.int32)
        mutal_person_mask = tf.cast(tf.clip_by_value((batch1_mask + batch2_mask), 0, 1), tf.float32)
        ground_truth_mask = 1 - (1 - mask) * mutal_person_mask

    else:
        ground_truth_mask = mask

    if invert_mask:
        ground_truth_mask = 1 - ground_truth_mask

    new_batch = batch1 * ground_truth_mask + batch2 * (1 - ground_truth_mask)

    return ground_truth_mask, new_batch


def cut_out(
    batch: tf.Tensor, invert_mask: bool = False, mask: Optional[tf.Tensor] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Performs the cutout operation of a batch of images.

    Args:
        batch (tf.Tensor): Batch of grid-shaped data of shape (`batch`, `height`, `width`, `channel`).
        invert_mask (bool, optional):  If true, the mask is inverted. 1->0 and 0->1. Defaults to False.
        mask (Optional[tf.Tensor], optional): Binary mask that requires same shape as `batch1` and `batch2`. If `None`
            mask is generated randomly. Defaults to None.

    Returns:
        mask: Actual mask that has been applied
        new_batch: Batch with applied cutout opperation
    """
    batch = tf.cast(batch, tf.float32)

    if mask is None:  # generate mask
        mask = _get_mask(shape=batch.shape)

    if invert_mask:
        mask = 1 - mask

    new_batch = batch * mask
    return mask, new_batch

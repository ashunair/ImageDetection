"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras.backend
from .dynamic import meshgrid
import  tensorflow as tf
import numpy as np


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.1, 0.1, 0.2, 0.2]

    widths  = boxes[:, :, 2] - boxes[:, :, 0]
    heights = boxes[:, :, 3] - boxes[:, :, 1]
    ctr_x   = boxes[:, :, 0] + 0.5 * widths
    ctr_y   = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0] * std[0] + mean[0]
    dy = deltas[:, :, 1] * std[1] + mean[1]
    dw = deltas[:, :, 2] * std[2] + mean[2]
    dh = deltas[:, :, 3] * std[3] + mean[3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w     = tf.keras.backend.exp(dw) * widths
    pred_h     = tf.keras.backend.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = tf.keras.backend.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=2)

    return pred_boxes


def shift(shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size
    """
    shift_x = (tf.range(0, shape[1], dtype=tf.keras.backend.floatx()) + tf.keras.backend.constant(0.5, dtype=tf.keras.backend.floatx())) * stride
    shift_y = (tf.range(0, shape[0], dtype=tf.keras.backend.floatx()) + tf.keras.backend.constant(0.5, dtype=tf.keras.backend.floatx())) * stride

    shift_x, shift_y = meshgrid(shift_x, shift_y)
    shift_x = tf.keras.backend.reshape(shift_x, [-1])
    shift_y = tf.keras.backend.reshape(shift_y, [-1])

    shifts = tf.keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = tf.keras.backend.transpose(shifts)
    number_of_anchors = tf.keras.backend.shape(anchors)[0]

    k = tf.keras.backend.int_shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = tf.keras.backend.cast(anchors, 'float32') + tf.keras.backend.cast(tf.keras.backend.reshape(shifts, [k, 1, 4]), 'float32')
    shifted_anchors = tf.keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors

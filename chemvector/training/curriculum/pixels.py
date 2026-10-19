"""Unchanged 768px RGB normalization shared by training and evaluation."""

import numpy as np
import torch


def image_tensor(image):
    if image.size != (768, 768) or image.mode != "RGB":
        raise ValueError("Unexpected input image shape/mode; no implicit conversion or resizing")
    return (torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255 - .5) / .5

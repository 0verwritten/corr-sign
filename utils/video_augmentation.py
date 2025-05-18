from typing import cast
from PIL import Image
from PIL.ImageOps import expand
import cv2

import torch
import random
import numbers
import numpy as np

class TemporalRescale:
    def __init__(self, temp_scaling=0.2, frame_interval=1):
        self.min_len = 32
        self.max_len = int(np.ceil(230 / frame_interval))
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        vid_len = len(clip)
        scaling_factor = self.L + (self.U - self.L) * np.random.random()
        new_len = max(self.min_len, min(int(vid_len * scaling_factor), self.max_len))
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        index = np.linspace(0, vid_len - 1, new_len).astype(np.int32)
        rescaled_clip = [clip[i] for i in index]
        return torch.stack([torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame for frame in rescaled_clip])

class ToTensor(object):
    def __call__(self, video):
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2)))
        return video

class NormalizeVideo:
    def __call__(self, clip):
        return (clip.float() / 127.5) - 1.0

class Resize:
    def __init__(self, rate=0.0, interp=cv2.INTER_LINEAR):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        if self.rate == 1.0:
            return clip
        scale = self.rate
        if isinstance(clip[0], np.ndarray):
            h, w = clip[0].shape[:2]
            nh, nw = int(h * scale), int(w * scale)
            return [
                cv2.resize(img, (nw, nh), interpolation=self.interpolation)
                for img in clip
            ]
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
            nw, nh = int(w * scale), int(h * scale)
            return [cast(Image.Image, img).resize((nw, nh), resample=Image.Resampling.BILINEAR) for img in clip]
        else:
            raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(clip[0])}")

class RandomCrop:
    """
    Randomly crops a region of the specified size from each frame of a video clip.
    Pads the frame if the crop size is larger than the frame size.

    Args:
        size (tuple[int, int] or int): Desired output size (height, width).
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size <= 0:
                raise ValueError("Size must be a positive number.")
            self.size = (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            self.size = tuple(size)
        else:
            raise ValueError(
                "Size must be a single positive int or a tuple of two ints (height, width)."
            )

    def __call__(self, clip):
        if not clip:
            raise ValueError("Clip is empty.")

        crop_h, crop_w = self.size
        first_frame = clip[0]

        if isinstance(first_frame, np.ndarray):
            im_h, im_w = first_frame.shape[:2]
        elif isinstance(first_frame, Image):
            im_w, im_h = first_frame.size
        else:
            raise TypeError(f"Unsupported frame type: {type(first_frame)}")

        # Compute padding if crop is larger than frame
        pad_h = max(crop_h - im_h, 0)
        pad_w = max(crop_w - im_w, 0)

        if pad_h > 0 or pad_w > 0:
            clip = [self._pad_frame(f, pad_h, pad_w) for f in clip]
            im_h += pad_h
            im_w += pad_w

        # Random crop position
        h1 = random.randint(0, im_h - crop_h)
        w1 = random.randint(0, im_w - crop_w)

        return [self._crop_frame(f, h1, w1, crop_h, crop_w) for f in clip]

    def _pad_frame(self, frame, pad_h, pad_w):
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if isinstance(frame, np.ndarray):
            return np.pad(
                frame,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif isinstance(frame, Image):
            return expand(
                frame, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
            )

    def _crop_frame(self, frame, top, left, height, width):
        if isinstance(frame, np.ndarray):
            return frame[top : top + height, left : left + width, :]
        elif isinstance(frame, Image.Image):
            return frame.crop((left, top, left + width, top + height))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        try:
            im_h, im_w, im_c = clip[0].shape
        except ValueError:
            print(clip[0].shape)
        new_h, new_w = self.size
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.0))
        left = int(round((im_w - new_w) / 2.0))
        return [img[top : top + new_h, left : left + new_w] for img in clip]


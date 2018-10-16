from __future__ import division
import torch
import math
import random
from PIL import Image
import numpy as np
import numbers
import types
import collections
import warnings
import cv2


def normalize(tensor, mean, std):
    """Normalize a ``torch.tensor``

    Args:
        tensor (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR

    Returns:
        Tensor: Normalized tensor.
    """

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()


def resize(img, mask, kpt, center, ratio):
    """Resize the ``numpy.ndarray`` and points as ratio.

    Args:
        img    (numpy.ndarray):   Image to be resized.
        mask   (numpy.ndarray):   Mask to be resized.
        kpt    (list):            Keypoints to be resized.
        center (list):            Center points to be resized.
        ratio  (tuple or number): the ratio to resize.

    Returns:
        numpy.ndarray: Resized image.
        numpy.ndarray: Resized mask.
        lists:         Resized keypoints.
        lists:         Resized center points.
    """

    if not (isinstance(ratio, numbers.Number) or (isinstance(ratio, collections.Iterable) and len(ratio) == 2)):
        raise TypeError('Got inappropriate ratio arg: {}'.format(ratio))

    h, w, _ = img.shape
    if w < 64:
        img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - w, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        mask = cv2.copyMakeBorder(mask, 0, 0, 0, 64 - w, cv2.BORDER_CONSTANT, value=(1, 1, 1))
        w = 64

    if isinstance(ratio, numbers.Number):

        num = len(kpt)
        length = len(kpt[0])
        for i in range(num):
            for j in range(length):
                kpt[i][j][0] *= ratio
                kpt[i][j][1] *= ratio
            center[i][0] *= ratio
            center[i][1] *= ratio

        return cv2.resize(img, (0, 0), fx=ratio, fy=ratio), cv2.resize(mask, (0, 0), fx=ratio, fy=ratio), kpt, center

    else:
        num = len(kpt)
        length = len(kpt[0])
        for i in range(num):
            for j in range(length):
                kpt[i][j][0] *= ratio[0]
                kpt[i][j][1] *= ratio[1]
            center[i][0] *= ratio[0]
            center[i][1] *= ratio[1]
        return np.ascontiguousarray(cv2.resize(img, (0, 0), fx=ratio[0], fy=ratio[1])), np.ascontiguousarray(
            cv2.resize(mask, (0, 0), fx=ratio[0], fy=ratio[1])), kpt, center


class RandomResized(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_min=0.8, scale_max=1.2):
        self.scale_min = scale_min
        self.scale_max = scale_max

    @staticmethod
    def get_params(img, scale_min, scale_max, scale):
        height, width, _ = img.shape

        ratio = random.uniform(scale_min, scale_max)
        ratio = ratio * 0.6 / scale

        return ratio

    def __call__(self, img, mask, kpt, center, scale):
        """
        Args:
            img     (numpy.ndarray): Image to be resized.
            mask    (numpy.ndarray): Mask to be resized.
            kpt     (list):          keypoints to be resized.
            center: (list):          center points to be resized.

        Returns:
            numpy.ndarray: Randomly resize image.
            numpy.ndarray: Randomly resize mask.
            list:          Randomly resize keypoints.
            list:          Randomly resize center points.
        """
        ratio = self.get_params(img, self.scale_min, self.scale_max, scale[0])

        return resize(img, mask, kpt, center, ratio)


class TestResized(object):
    """Resize the given numpy.ndarray to the size for test.

    Args:
        size: the size to resize.
    """

    def __init__(self, size):
        assert (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):

        height, width, _ = img.shape

        return (output_size[0] * 1.0 / width, output_size[1] * 1.0 / height)

    def __call__(self, img, mask, kpt, center):
        """
        Args:
            img     (numpy.ndarray): Image to be resized.
            mask    (numpy.ndarray): Mask to be resized.
            kpt     (list):          keypoints to be resized.
            center: (list):          center points to be resized.

        Returns:
            numpy.ndarray: Randomly resize image.
            numpy.ndarray: Randomly resize mask.
            list:          Randomly resize keypoints.
            list:          Randomly resize center points.
        """
        ratio = self.get_params(img, self.size)

        return resize(img, mask, kpt, center, ratio)


def crop(img, mask, kpt, center, offset_left, offset_up, w, h):
    num = len(kpt)
    length = len(kpt[0])

    for x in range(num):
        for y in range(length):
            kpt[x][y][0] -= offset_left
            kpt[x][y][1] -= offset_up
        center[x][0] -= offset_left
        center[x][1] -= offset_up

    height, width, _ = img.shape
    mask = mask.reshape((height, width))

    new_img = np.empty((h, w, 3), dtype=np.float32)
    new_img.fill(128)

    new_mask = np.empty((h, w), dtype=np.float32)
    new_mask.fill(1)

    st_x = 0
    ed_x = w
    st_y = 0
    ed_y = h
    or_st_x = offset_left
    or_ed_x = offset_left + w
    or_st_y = offset_up
    or_ed_y = offset_up + h

    if offset_left < 0:
        st_x = -offset_left
        or_st_x = 0
    if offset_left + w > width:
        ed_x = width - offset_left
        or_ed_x = width
    if offset_up < 0:
        st_y = -offset_up
        or_st_y = 0
    if offset_up + h > height:
        ed_y = height - offset_up
        or_ed_y = height

    new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()
    new_mask[st_y: ed_y, st_x: ed_x] = mask[or_st_y: or_ed_y, or_st_x: or_ed_x].copy()

    return np.ascontiguousarray(new_img), np.ascontiguousarray(new_mask), kpt, center


class RandomCrop(object):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, center_perturb_max=40):
        assert isinstance(size, numbers.Number)
        self.size = (int(size), int(size))  # (w, h)
        self.center_perturb_max = center_perturb_max

    @staticmethod
    def get_params(img, center, output_size, center_perturb_max):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img                (numpy.ndarray): Image to be cropped.
            center             (list):          the center of main person.
            output_size        (tuple):         Expected output size of the crop.
            center_perturb_max (int):           the max perturb size.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        ratio_x = random.uniform(0, 1)
        ratio_y = random.uniform(0, 1)
        x_offset = int((ratio_x - 0.5) * 2 * center_perturb_max)
        y_offset = int((ratio_y - 0.5) * 2 * center_perturb_max)
        center_x = center[0][0] + x_offset
        center_y = center[0][1] + y_offset

        return int(round(center_x - output_size[0] / 2)), int(round(center_y - output_size[1] / 2))

    def __call__(self, img, mask, kpt, center):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.
            mask (numpy.ndarray): Mask to be cropped.
            kpt (list): keypoints to be cropped.
            center (list): center points to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
            numpy.ndarray: Cropped mask.
            list:          Cropped keypoints.
            list:          Cropped center points.
        """

        offset_left, offset_up = self.get_params(img, center, self.size, self.center_perturb_max)

        return crop(img, mask, kpt, center, offset_left, offset_up, self.size[0], self.size[1])




def hflip(img, mask, kpt, center):
    height, width, _ = img.shape
    mask = mask.reshape((height, width, 1))

    img = img[:, ::-1, :]
    mask = mask[:, ::-1, :]

    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] <= 1:
                kpt[i][j][0] = width - 1 - kpt[i][j][0]
        center[i][0] = width - 1 - center[i][0]

    swap_pair = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]
    for x in swap_pair:
        for i in range(num):
            temp_point = kpt[i][x[0] - 1]
            kpt[i][x[0] - 1] = kpt[i][x[1] - 1]
            kpt[i][x[1] - 1] = temp_point

    return np.ascontiguousarray(img), np.ascontiguousarray(mask), kpt, center


class RandomHorizontalFlip(object):
    """Random horizontal flip the image.

    Args:
        prob (number): the probability to flip.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask, kpt, center):
        """
        Args:
            img    (numpy.ndarray): Image to be flipped.
            mask   (numpy.ndarray): Mask to be flipped.
            kpt    (list):          Keypoints to be flipped.
            center (list):          Center points to be flipped.

        Returns:
            numpy.ndarray: Randomly flipped image.
            list: Randomly flipped points.
        """
        if random.random() < self.prob:
            return hflip(img, mask, kpt, center)
        return img, mask, kpt, center


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Mytransforms.Compose([
        >>>     Mytransforms.CenterCrop(10),
        >>>     Mytransforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpt, bb, ex_kpt, ex_bb):

        for t in self.transforms:
            if isinstance(t, Crop):
                img, kpt, ex_kpt = t(img, kpt, bb, ex_kpt, ex_bb)
            else:
                img, kpt, ex_kpt = t(img, kpt, ex_kpt)

        return img, kpt, ex_kpt


class Crop(object):
    """Crop the given numpy.ndarray .

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        assert isinstance(size, numbers.Number)
        self.size = (int(size), int(size))  # (w, h)

    def __call__(self, img, kpt, bb, ex_kpt, ex_bb):

        # bb[0:4] = [height1, height2, width1, width2]
        # kpt items[0:4] = [height, width, is_visible(0 for True and 2 for False), id(0-13)]

        # print(3)
        # width, height = img.size
        height, width = img.shape[0], img.shape[1]
        bb[0] = int(bb[0])
        bb[1] = int(bb[1])
        bb[2] = int(bb[2])
        bb[3] = int(bb[3])
        size1 = bb[1] - bb[0]
        # print('width height', width, height)
        # print('bb', bb[0], bb[1], bb[2], bb[3])
        # print('size1', size1)
        img0 = Image.new('RGB', (size1, size1), (128, 128, 128))
        img0 = np.ones((size1, size1, 3))
        img0 *= 128
        if bb[0] < 0:
            if bb[1] > height:
                if bb[2] < 0:
                    if bb[3] > width:
                        # img1 = img.crop((0, 0, width, height))
                        img1 = img[0:height, 0:width, :]
                        # img0.paste(img1, (0 - bb[2], 0 - bb[0], width - bb[2], height - bb[0]))
                        img0[0 - bb[0]:height - bb[0], 0 - bb[2]:width - bb[2], :] = img1
                    else:
                        # img1 = img.crop((0, 0, bb[3], height))
                        img1 = img[0:height, 0:bb[3], :]
                        # img0.paste(img1, (0 - bb[2], 0 - bb[0], bb[3] - bb[2], height - bb[0]))
                        img0[0 - bb[0]:height - bb[0], 0 - bb[2]:bb[3] - bb[2], :] = img1
                else:
                    if bb[3] > width:
                        # img1 = img.crop((bb[2], 0, width, height))
                        img1 = img[0:height, bb[2]:width, :]
                        # img0.paste(img1, (0, 0 - bb[0], width - bb[2], height - bb[0]))
                        img0[0 - bb[0]:height - bb[0], 0:width - bb[2], :] = img1
                    else:
                        # img1 = img.crop((bb[2], 0, bb[3], height))
                        img1 = img[0:height, bb[2]:bb[3], :]
                        # img0.paste(img1, (0, 0 - bb[0], bb[3] - bb[2], height - bb[0]))
                        img0[0 - bb[0]:height - bb[0], 0:bb[3] - bb[2], :] = img1
            else:
                if bb[2] < 0:
                    if bb[3] > width:
                        # img1 = img.crop((0, 0, width, bb[1]))
                        img1 = img[0:bb[1], 0:width, :]
                        # img0.paste(img1, (0 - bb[2], 0 - bb[0], width - bb[2], bb[1] - bb[0]))
                        img0[0 - bb[0]:bb[1] - bb[0], 0 - bb[2]:width - bb[2], :] = img1
                    else:
                        # img1 = img.crop((0, 0, bb[3], bb[1]))
                        img1 = img[0:bb[1], 0:bb[3], :]
                        # img0.paste(img1, (0 - bb[2], 0 - bb[0], bb[3] - bb[2], bb[1] - bb[0]))
                        img0[0 - bb[0]:bb[1] - bb[0], 0 - bb[2]:bb[3] - bb[2], :] = img1
                else:
                    if bb[3] > width:
                        # img1 = img.crop((bb[2], 0, width, bb[1]))
                        img1 = img[0:bb[1], bb[2]:width, :]
                        # img0.paste(img1, (0, 0 - bb[0], width - bb[2], bb[1] - bb[0]))
                        img0[0 - bb[0]:bb[1] - bb[0], 0:width - bb[2], :] = img1
                    else:
                        # img1 = img.crop((bb[2], 0, bb[3], bb[1]))
                        img1 = img[0:bb[1], bb[2]:bb[3], :]
                        # img0.paste(img1, (0, 0 - bb[0], bb[3] - bb[2], bb[1] - bb[0]))
                        img0[0 - bb[0]:bb[1] - bb[0], 0:bb[3] - bb[2], :] = img1
        else:
            if bb[1] > height:
                if bb[2] < 0:
                    if bb[3] > width:
                        # img1 = img.crop((0, bb[0], width, height))
                        img1 = img[bb[0]:height, 0:width, :]
                        # img0.paste(img1, (0 - bb[2], 0, width - bb[2], height - bb[0]))
                        img0[0:height - bb[0], 0 - bb[2]:width - bb[2], :] = img1
                    else:
                        # img1 = img.crop((0, bb[0], bb[3], height))
                        img1 = img[bb[0]:height, 0:bb[3], :]
                        # img0.paste(img1, (0 - bb[2], 0, bb[3] - bb[2], height - bb[0]))
                        img0[0:height - bb[0], 0 - bb[2]:bb[3] - bb[2], :] = img1
                else:
                    if bb[3] > width:
                        # img1 = img.crop((bb[2], bb[0], width, height))
                        img1 = img[bb[0]:height, bb[2]:width, :]
                        # img0.paste(img1, (0, 0, width - bb[2], height - bb[0]))
                        img0[0:height - bb[0], 0:width - bb[2], :] = img1
                    else:
                        # img1 = img.crop((bb[2], bb[0], bb[3], height))
                        img1 = img[bb[0]:height, bb[2]:bb[3], :]
                        
                        try:
                            # img0.paste(img1, (0, 0, bb[3] - bb[2], height - bb[0]))
                            img0[0:height - bb[0], 0:bb[3] - bb[2], :] = img1
                        except:
                            print('bb', bb[0], bb[1], bb[2], bb[3])
                            print('crop', bb[2], bb[0], bb[3], height)
                            print('paste', 0, 0, bb[3] - bb[2], height - bb[0])
                            print('width height', width, height)
                            print('size1', size1)
            else:
                if bb[2] < 0:
                    if bb[3] > width:
                        # img1 = img.crop((0, bb[0], width, bb[1]))
                        img1 = img[bb[0]:bb[1], 0:width, :]
                        # img0.paste(img1, (0 - bb[2], 0, width - bb[2], bb[1] - bb[0]))
                        img0[0:bb[1] - bb[0], 0 - bb[2]:width - bb[2], :] = img1
                    else:
                        # img1 = img.crop((0, bb[0], bb[3], bb[1]))
                        img1 = img[bb[0]:bb[1], 0:bb[3], :]
                        # img0.paste(img1, (0 - bb[2], 0, bb[3] - bb[2], bb[1] - bb[0]))
                        img0[0:bb[1] - bb[0], 0 - bb[2]:bb[3] - bb[2], :] = img1
                else:
                    if bb[3] > width:
                        # img1 = img.crop((bb[2], bb[0], width, bb[1]))
                        img1 = img[bb[0]:bb[1], bb[2]:width, :]
                        # img0.paste(img1, (0, 0, width - bb[2], bb[1] - bb[0]))
                        img0[0:bb[1] - bb[0], 0:width - bb[2], :] = img1
                    else:
                        # img1 = img.crop((bb[2], bb[0], bb[3], bb[1]))
                        img1 = img[bb[0]:bb[1], bb[2]:bb[3], :]
                        # img0.paste(img1, (0, 0, bb[3] - bb[2], bb[1] - bb[0]))
                        img0[0:bb[1] - bb[0], 0:bb[3] - bb[2], :] = img1

        # print(4)
        # img2 = img0.resize(self.size)
        img2 = cv2.resize(img0, self.size)
        ratio = self.size[0] * 1.0 / size1
        for items in kpt:
            if items[2] == 0:
                items[0] = int((items[0] - bb[0]) * ratio)
                items[1] = int((items[1] - bb[2]) * ratio)
        ex_ratio = self.size[0] * 1.0 / (ex_bb[1] - ex_bb[0])
        for items in ex_kpt:
            if items[2] == 0:
                items[0] = int((items[0] - ex_bb[0]) * ex_ratio)
                items[1] = int((items[1] - ex_bb[2]) * ex_ratio)
        # print(5)
        return img2, kpt, ex_kpt


class RandomTranspose(object):
    """Randomly Transpose the given PIL.Image.

    """

    def __init__(self):
        pass

    def __call__(self, img, kpt, ex_kpt):

        # kpt items[0:4] = [height, width, is_visible(0 for True and 2 for False), id(0-13)]

        self.num = random.random()
        width, height = img.size
        if self.num < 0.125:
            pass
        elif self.num < 0.25:
            img = img.transpose(Image.ROTATE_90)
            for items in kpt:
                if items[2] == 0:
                    x = items[0]
                    y = items[1]
                    items[0] = width - y
                    items[1] = x
            for items in ex_kpt:
                if items[2] == 0:
                    x = items[0]
                    y = items[1]
                    items[0] = width - y
                    items[1] = x
        elif self.num < 0.375:
            img = img.transpose(Image.ROTATE_180)
            for items in kpt:
                if items[2] == 0:
                    items[0] = height - items[0]
                    items[1] = width - items[1]
            for items in ex_kpt:
                if items[2] == 0:
                    items[0] = height - items[0]
                    items[1] = width - items[1]
        elif self.num < 0.5:
            img = img.transpose(Image.ROTATE_270)
            for items in kpt:
                if items[2] == 0:
                    x = items[0]
                    y = items[1]
                    items[0] = y
                    items[1] = height - x
            for items in ex_kpt:
                if items[2] == 0:
                    x = items[0]
                    y = items[1]
                    items[0] = y
                    items[1] = height - x
        elif self.num < 0.625:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for items in kpt:
                if items[2] == 0:
                    items[0] = items[0]
                    items[1] = width - items[1]
            for items in ex_kpt:
                if items[2] == 0:
                    items[0] = items[0]
                    items[1] = width - items[1]
        elif self.num < 0.75:
            img = img.transpose(Image.ROTATE_90)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for items in kpt:
                if items[2] == 0:
                    x = items[0]
                    y = items[1]
                    items[0] = width - y
                    items[1] = height - x
            for items in ex_kpt:
                if items[2] == 0:
                    x = items[0]
                    y = items[1]
                    items[0] = width - y
                    items[1] = height - x
        elif self.num < 0.875:
            img = img.transpose(Image.ROTATE_180)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for items in kpt:
                if items[2] == 0:
                    items[0] = height - items[0]
                    items[1] = items[1]
            for items in ex_kpt:
                if items[2] == 0:
                    items[0] = height - items[0]
                    items[1] = items[1]
        else:
            img = img.transpose(Image.ROTATE_270)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for items in kpt:
                if items[2] == 0:
                    x = items[0]
                    y = items[1]
                    items[0] = y
                    items[1] = x
            for items in ex_kpt:
                if items[2] == 0:
                    x = items[0]
                    y = items[1]
                    items[0] = y
                    items[1] = x

        return img, kpt, ex_kpt


def rotate(img, kpt, ex_kpt, degree):
    """Rotate the ``numpy.ndarray`` and points as degree.

    Args:
        img     (numpy.ndarray): Image to be rotated.
        kpt     (list):          Keypoints to be rotated.
        ex_kpt  (list):          Ex keypoints to be rotated.
        degree  (number):        the degree to rotate.

    Returns:
        numpy.ndarray: Rotated image.
        list:          Rotated keypoints.
        list:          Rotated ex keypoints.
    """

    height, width, _ = img.shape
    size = max(height, width)

    img_center = (width / 2.0, height / 2.0)

    rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)
    cos_val = np.abs(rotateMat[0, 0])
    sin_val = np.abs(rotateMat[0, 1])
    new_width = int(height * sin_val + width * cos_val)
    new_height = int(height * cos_val + width * sin_val)
    new_size = max(new_width, new_height)
    rotateMat[0, 2] += (new_size / 2.) - img_center[0]
    rotateMat[1, 2] += (new_size / 2.) - img_center[1]

    img = cv2.warpAffine(img, rotateMat, (new_size, new_size), borderValue=(128, 128, 128))

    img = np.ascontiguousarray(img)

    for items in kpt:
        if items[2] == 0:
            x = items[0]
            y = items[1]
            p = np.array([y, x, 1])
            p = rotateMat.dot(p)
            items[0] = int(p[1])
            items[1] = int(p[0])
    for items in ex_kpt:
        if items[2] == 0:
            x = items[0]
            y = items[1]
            p = np.array([y, x, 1])
            p = rotateMat.dot(p)
            items[0] = int(p[1])
            items[1] = int(p[0])

    img = cv2.resize(img, (size, size))
    ratio = size * 1.0 / new_size
    for items in kpt:
        if items[2] == 0:
            items[0] = int(items[0] * ratio)
            items[1] = int(items[1] * ratio)
    for items in ex_kpt:
        if items[2] == 0:
            items[0] = int(items[0] * ratio)
            items[1] = int(items[1] * ratio)

    return img, kpt, ex_kpt


class RandomRotate(object):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree):
        assert isinstance(max_degree, numbers.Number)
        self.max_degree = max_degree

    @staticmethod
    def get_params(max_degree):
        """Get parameters for ``rotate`` for a random rotate.

        Returns:
            number: degree to be passed to ``rotate`` for random rotate.
        """
        degree = random.uniform(-max_degree, max_degree)

        return degree

    def __call__(self, img, kpt, ex_pkt):
        """
        Args:
            img    (numpy.ndarray): Image to be rotated.
            mask   (numpy.ndarray): Mask to be rotated.
            kpt    (list):          Keypoints to be rotated.
            center (list):          Center points to be rotated.

        Returns:
            numpy.ndarray: Rotated image.
            list:          Rotated key points.
        """
        degree = self.get_params(self.max_degree)

        return rotate(img, kpt, ex_pkt, degree)


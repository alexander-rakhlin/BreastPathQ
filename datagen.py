from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as Soup
import cv2
from itertools import islice, chain

from keras import backend as K
import threading
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    RandomRotate90,
    GridDistortion,
    HueSaturationValue,
    RandomGamma,
    ShiftScaleRotate,
)

R = 12
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]    # R, G, B
CLASSES = (0, 1, 2, 3)
MALIGNANT = ("IDC", "ILC", "Muc C", "DCIS 1", "DCIS 2", "DCIS 3", "MC- E", "MC - C", "MC - M")
NORMAL = ("normal", "UDH", "ADH")
LYMPHOCYTE = ("TIL-E", "TIL-S")
CLASS_DICT = {}
CLASS_DICT.update(dict.fromkeys(MALIGNANT, 0))
CLASS_DICT.update(dict.fromkeys(NORMAL, 1))
CLASS_DICT.update(dict.fromkeys(LYMPHOCYTE, 2))


def read_coords(fl, verbose=False):
    soup = Soup(open(fl), "lxml")
    elements = soup.findAll("graphic")
    cls = []
    xy = []
    for element in elements:
        if element.attrs["description"] not in CLASS_DICT:
            if verbose:
                print(fl.name, element.attrs["description"])
            continue
        points = element.findAll("point")
        xy.extend(tuple(int(p) for p in point.contents[0].split(",")) for point in points)
        cls.extend([CLASS_DICT[element.attrs["description"]]] * len(points))
    return xy, cls


def norm_fun(x):
    return (x / 255. - 0.5) * 2


def renorm_fun(x):
    return np.uint8(np.clip((x / 2 + 0.5) * 255., 0, 255))


def read_img_mask(img_path, sz, downscale=1):
    key_path = img_path.parent / img_path.name.replace("_crop.tif", "_key.xml")
    xy, cls = read_coords(key_path)
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), np.uint8)
    for coord, cl in zip(xy, cls):
        cv2.circle(mask, coord, R, COLORS[cl], -1)

    if downscale != 1:
        scale = 1 / downscale
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    H, W = sz
    h_ = max(H, h)
    w_ = max(W, w)
    if (H > h) or (W > w):
        img_ = np.zeros((h_, w_, 3), np.uint8)
        mask_ = np.zeros((h_, w_, 3), np.uint8)
        img_[(h_ - h) // 2: (h_ - h) // 2 + h, (w_ - w) // 2: (w_ - w) // 2 + w] = img
        mask_[(h_ - h) // 2: (h_ - h) // 2 + h, (w_ - w) // 2: (w_ - w) // 2 + w] = mask
    else:
        img_ = img
        mask_ = mask

    x_coord = np.random.randint(w_ - W) if w_ > W else 0
    y_coord = np.random.randint(h_ - H) if h_ > H else 0
    img = img_[y_coord: y_coord + H, x_coord: x_coord + W]
    mask = mask_[y_coord: y_coord + H, x_coord: x_coord + W]
    return img, mask


def read_img(img_path, sz, downscale=1):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    if downscale != 1:
        scale = 1 / downscale
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    H, W = sz
    h_ = max(H, h)
    w_ = max(W, w)
    if (H > h) or (W > w):
        img_ = np.zeros((h_, w_, 3), np.uint8)
        img_[(h_ - h) // 2: (h_ - h) // 2 + h, (w_ - w) // 2: (w_ - w) // 2 + w] = img
    else:
        img_ = img

    x_coord = np.random.randint(w_ - W) if w_ > W else 0
    y_coord = np.random.randint(h_ - H) if h_ > H else 0
    img = img_[y_coord: y_coord + H, x_coord: x_coord + W]
    return img


def mask_from_rgb(x):
    x = np.round(x / 255.)
    x_cl = np.argmax(x, axis=-1)
    x_cl[x.sum(axis=-1) == 0] = CLASSES[-1]
    x = np.zeros(x.shape[:2] + (len(CLASSES),), dtype=K.floatx())
    for c in CLASSES:
        x[x_cl == c, c] = 1
    return x


def mask_to_rgb(x):
    return (x[..., :-1] * 255).astype(np.uint8)


def train_transform(sz, downscale=1, p=1):
    augmentation = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        # GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # RandomGamma(p=0.9, gamma_limit=(80, 150)),
        HueSaturationValue(p=0.9, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
    ], p=p)

    def transform_fun(img_path):
        img, mask = read_img_mask(img_path, sz, downscale=downscale)
        data = {"image": img, "mask": mask}
        augmented = augmentation(**data)
        img, mask = augmented["image"], augmented["mask"]

        img = norm_fun(img)
        mask = mask_from_rgb(mask)
        return img, mask

    return transform_fun


def val_transform(sz, downscale=1, p=1):
    augmentation = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
    ], p=p)

    def transform_fun(img_path):
        img, mask = read_img_mask(img_path, sz, downscale=downscale)
        data = {"image": img, "mask": mask}
        augmented = augmentation(**data)
        img, mask = augmented["image"], augmented["mask"]

        img = norm_fun(img)
        mask = mask_from_rgb(mask)
        return img, mask

    return transform_fun


def train_transform_regr(sz, downscale=1, p=1):
    augmentation = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomGamma(p=0.9, gamma_limit=(80, 150)),
        HueSaturationValue(p=0.9, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
    ], p=p)

    def transform_fun(img_path, n=None):
        img = read_img(img_path, sz, downscale=downscale)
        data = {"image": img}
        augmented = augmentation(**data)
        img = augmented["image"]

        img = norm_fun(img)
        return img

    return transform_fun


def val_transform_regr(sz, downscale=1, p=1):
    augmentation = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
    ], p=p)

    def transform_fun(img_path, n=None):
        img = read_img(img_path, sz, downscale=downscale)
        data = {"image": img}
        augmented = augmentation(**data)
        img = augmented["image"]

        img = norm_fun(img)
        return img

    return transform_fun


def inference_transform_regr(sz, downscale=1):

    def transform_fun(img_path, n=None):
        img = read_img(img_path, sz, downscale=downscale)
        img = [img, img[::-1]][(n // 4) % 2]
        img = np.rot90(img, k=n % 4)
        img = norm_fun(img)
        return img
    return transform_fun


class ImageIterator(object):
    def __init__(self, image_list, tile_sz,
                 n_classes=4, transform_fun=lambda *args: args,
                 batch_size=16, shuffle=True, seed=None, verbose=False,
                 output_ids=False, gen_id=""):
        self.image_list = image_list
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.transform_fun = transform_fun
        self.output_ids = output_ids
        self.gen_id = gen_id
        self.seed = seed
        self.verbose = verbose

        if isinstance(tile_sz, (tuple, list)):
            self.h, self.w = tile_sz
        else:
            self.h = self.w = tile_sz
        if K.image_data_format() == "channels_last":
            self.image_shape = (self.h, self.w, 3)
            self.mask_shape = (self.h, self.w, self.n_classes)
        else:
            self.image_shape = (3, self.h, self.w)
            self.mask_shape = (self.n_classes, self.h, self.w)

        self.index_generator = self._flow_index(len(self.image_list))

    def next(self):

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        with self.lock:
            image_ids, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((current_batch_size,) + self.mask_shape, dtype=K.floatx())
        for b, image_id in enumerate(image_ids):
            image_path = self.image_list[image_id]
            image, mask = self.transform_fun(image_path)
            batch_x[b] = image if K.image_data_format() == "channels_last" else np.moveaxis(image, -1, 0)
            batch_y[b] = mask if K.image_data_format() == "channels_last" else np.moveaxis(mask, -1, 0)

        if self.output_ids:
            return batch_x, batch_y, [self.image_list[i] for i in image_ids]
        else:
            return batch_x, batch_y

    def _flow_index(self, n):
        # Ensure self.batch_index is 0.
        self.batch_index = 0
        image_ids = list(range(n))
        while 1:
            if self.seed is None:
                random_seed = None
            else:
                random_seed = self.seed + self.total_batches_seen
            if self.batch_index == 0:
                if self.verbose:
                    print("\n************** New epoch. Generator", self.gen_id, "*******************\n")
                if self.shuffle:
                    np.random.RandomState(random_seed).shuffle(image_ids)
            current_index = (self.batch_index * self.batch_size) % n
            if n > current_index + self.batch_size:
                current_batch_size = self.batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (image_ids[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class ImageIteratorRegr(object):
    def __init__(self, image_list, labels_csv, tile_sz,
                 transform_fun=lambda *args: args,
                 batch_size=16, shuffle=True, seed=None, verbose=False,
                 output_ids=False, gen_id=""):
        self.image_list = image_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.transform_fun = transform_fun
        self.output_ids = output_ids
        self.gen_id = gen_id
        self.seed = seed
        self.verbose = verbose
        self.id_seen = dict.fromkeys(range(len(self.image_list)), 0)

        if labels_csv:
            labels_df = pd.read_csv(labels_csv)
            p_col = "y" if "y" in labels_df.columns else "p"
            name_cellularity = zip(labels_df["slide"].map(str) + "_" + labels_df["rid"].map(str), labels_df[p_col])
            self.label_dict = {name: cellularity for name, cellularity in name_cellularity}
        else:
            self.label_dict = None

        if isinstance(tile_sz, (tuple, list)):
            self.h, self.w = tile_sz
        else:
            self.h = self.w = tile_sz
        if K.image_data_format() == "channels_last":
            self.image_shape = (self.h, self.w, 3)
        else:
            self.image_shape = (3, self.h, self.w)

        self.index_generator = self._flow_index(len(self.image_list))

    def next(self):

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        with self.lock:
            image_ids, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        if self.label_dict:
            batch_y = np.zeros((current_batch_size, 1), dtype=K.floatx())
        else:
            batch_y = [None] * current_batch_size
        for b, image_id in enumerate(image_ids):
            image_path = self.image_list[image_id]
            image = self.transform_fun(image_path, self.id_seen[image_id])
            self.id_seen[image_id] += 1
            batch_x[b] = image if K.image_data_format() == "channels_last" else np.moveaxis(image, -1, 0)
            if self.label_dict:
                batch_y[b] = self.label_dict[image_path.stem]

        if self.output_ids:
            return batch_x, batch_y, [self.image_list[i] for i in image_ids]
        else:
            return batch_x, batch_y

    def _flow_index(self, n):
        # Ensure self.batch_index is 0.
        self.batch_index = 0
        image_ids = list(range(n))
        while 1:
            if self.seed is None:
                random_seed = None
            else:
                random_seed = self.seed + self.total_batches_seen
            if self.batch_index == 0:
                if self.verbose:
                    print("\n************** New epoch. Generator", self.gen_id, "*******************\n")
                if self.shuffle:
                    np.random.RandomState(random_seed).shuffle(image_ids)
            current_index = (self.batch_index * self.batch_size) % n
            if n > current_index + self.batch_size:
                current_batch_size = self.batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (image_ids[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


if __name__ == "__main__":
    sz = (256, 256)
    downscale = 2
    root = Path("data/datasets")

    image_list = sorted((root / "cells").glob("*_crop.tif"), key=lambda s: int(s.name.split("_")[0]))
    # image_list = [root / "cells" / "17_Region 60_crop.tif"]

    # batch_iterator = ImageIterator(image_list, sz, transform_fun=train_transform(sz, downscale=downscale),
    #                                batch_size=3, output_ids=True, shuffle=False)
    # x, y, ids = zip(*islice(batch_iterator, 10))
    # x = np.concatenate(x)
    # x = renorm_fun(x)
    # y = np.concatenate(y)
    # if K.image_data_format() == "channels_first":
    #     x = np.moveaxis(x, 1, -1)
    #     y = np.moveaxis(x, 1, -1)
    # ids = chain.from_iterable(ids)
    #
    # for x_, y_, ids_ in zip(x, y, ids):
    #     img = np.zeros((x_.shape[0], x_.shape[1] * 2, 3), dtype=np.uint8)
    #     img[:, :x_.shape[1]] = x_
    #     img[:, x_.shape[1]:] = mask_to_rgb(y_)
    #     cv2.imshow("/".join(ids_.parts[-2:]), img)
    #     cv2.waitKey()

    # for img_path in image_list:
    #     key_path = img_path.parent / img_path.name.replace("_crop.tif", "_key.xml")
    #     img_fit, mask_fit = read_img_mask(img_path, sz, downscale=downscale)
    #     cv2.imshow(key_path.name, np.hstack((img_fit, mask_fit)))
    #     cv2.waitKey()

    image_list = sorted((root / "train").glob("*.tif"), key=lambda s: int(s.stem.split("_")[0]))
    image_list = image_list[:2]
    batch_iterator = ImageIteratorRegr(image_list, root / "train_labels.csv", sz,
                                       transform_fun=inference_transform_regr(sz, downscale=downscale),
                                       batch_size=3, output_ids=True, shuffle=False)
    x, y, ids = zip(*islice(batch_iterator, 15))
    x = np.concatenate(x)
    x = renorm_fun(x)
    y = np.concatenate(y)
    if K.image_data_format() == "channels_first":
        x = np.moveaxis(x, 1, -1)
    ids = chain.from_iterable(ids)

    for x_, y_, ids_ in zip(x, y, ids):
        cv2.imshow("{} -> {}".format(ids_.stem, y_), x_)
        cv2.waitKey()

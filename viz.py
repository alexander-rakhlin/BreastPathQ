from datagen import ImageIterator
from datagen import train_transform
from models import uResNet34
from pathlib import Path
import cv2
import numpy as np
from datagen import renorm_fun
from skimage.feature import blob_log
import re
from itertools import islice

MODEL_ROOT = Path("models")
DATA_ROOT = Path("data/datasets")
BATCH_SZ = 8


def visualize(im_list, weights, channel=2, save_path=None):
    parse = re.match("(.+).sz([0-9]+)x([0-9]+).*z([0-9]+).", weights.stem)
    model_name = parse[1].replace("regr", "").replace("Xception", "")
    size = int(parse[2]), int(parse[3])
    downscale = int(parse[4])

    model_class = globals()[model_name]

    model = model_class(input_size=size, weights=weights)
    train_iterator = ImageIterator(im_list, size,
                                   transform_fun=train_transform(size, downscale=downscale),
                                   batch_size=BATCH_SZ, shuffle=False, output_ids=True)
    for x, y, ids in islice(train_iterator, len(im_list)):
        activations = model.predict_on_batch(x)
        for f_, x_, y_, ids_ in zip(activations, x, y, ids):
            h = x_.shape[0]
            w = x_.shape[1]
            pad = 3
            img = np.ones((h, w * 4 + pad * 3, 3), dtype=np.uint8) * 255
            img[:, :w] = renorm_fun(x_)
            img[:, w + pad: w * 2 + pad] = y_[..., [channel]] * 255
            img[:, w * 2 + pad * 2: w * 3 + pad * 2] = f_[..., [channel]] * 255

            blobs = blob_log(f_[..., channel], threshold=0.2, min_sigma=1, max_sigma=4, num_sigma=4, overlap=1)
            mask = np.zeros(x_.shape, np.uint8)
            for y_coord, x_coord, _ in blobs:
                x_coord, y_coord = int(x_coord), int(y_coord)
                cv2.circle(mask, (x_coord, y_coord), 4, (255, 255, 255), -1)
            img[:, w * 3 + pad * 3:] = mask

            if save_path:
                cv2.imwrite(save_path + "/" + ids_.stem + ".png", img)
            # cv2.imshow("{}".format(ids_.stem), img)
            # cv2.waitKey()


if __name__ == "__main__":
    weights = MODEL_ROOT / "uResNet34.sz256x256j0.005z2.520-0.678.set2.hdf5"
    image_list = sorted((DATA_ROOT / "cells").glob("*_crop.tif"), key=lambda s: int(s.name.split("_")[0]))
    train_test_split = int(len(image_list) * 0.75)
    train_list = image_list[:train_test_split]
    validation_list = image_list[train_test_split:]
    visualize(train_list, weights, channel=2, save_path="images/2")

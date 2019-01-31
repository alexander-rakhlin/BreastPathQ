# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from keras import backend as K
from models import uResNet34regr, uResNet34m46regr, uResNet34Xceptionregr
import re
from datagen import norm_fun, read_img
from utils import metrics, save_predictions
import pandas as pd

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    RandomRotate90,
    HueSaturationValue,
    RandomGamma,
)

ROOT = Path("data/datasets")
MODEL_ROOT = Path("models")
BATCH_SZ = 8
N_AUG = 20
MODELS = (
    # set 1
    # MODEL_ROOT / "uResNet34regr.sz512x512z1.50-0.016.hdf5",
    # MODEL_ROOT / "uResNet34m46regr.sz512x512z1.197-0.017.hdf5",
    # MODEL_ROOT / "uResNet34regr.sz256x256z2.80-0.014.hdf5",
    # MODEL_ROOT / "uResNet34m46regr.sz256x256z2.34-0.014.hdf5",
    # # set 2
    # MODEL_ROOT / "uResNet34m46regr.sz512x512z1.167-0.015.set2.hdf5",
    # MODEL_ROOT / "uResNet34regr.sz512x512z1.23-0.016.set2.hdf5",
    # MODEL_ROOT / "uResNet34m46regr.sz256x256z2.91-0.011.set2.hdf5",
    # MODEL_ROOT / "uResNet34regr.sz256x256z2.169-0.012.set2.hdf5",

    # Semi
    MODEL_ROOT / "uResNet34Xceptionregr.sz256x256z2.155-0.017.semi.set1.hdf5",
    MODEL_ROOT / "uResNet34regr.sz256x256z2.109-0.016.semi.S1.set1.hdf5",
    MODEL_ROOT / "uResNet34regr.sz256x256z2.192-0.017.semi.S2.set1.hdf5",
    MODEL_ROOT / "uResNet34regr.sz256x256z2.113-0.016.semi.S1.set2.hdf5",
    MODEL_ROOT / "uResNet34regr.sz256x256z2.107-0.018.semi.S2.set2.hdf5",
)


def make_tta_(img):
    return np.stack(np.rot90(img_, k=i) for img_ in [img, img[::-1]] for i in range(4))


def make_tta(img, p=1):
    augmentation = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomGamma(p=0.9, gamma_limit=(80, 150)),
        HueSaturationValue(p=0.9, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
    ], p=p)

    batch = []
    data = {"image": img}
    for _ in range(N_AUG):
        augmented = augmentation(**data)
        batch.append(augmented["image"])
    return np.array(batch)


def predict(image_list, verbose=True):
    predictions = {s.stem: [] for s in image_list}
    for model_path in MODELS:
        print("Model", model_path.stem)
        parse = re.match("(.+).sz([0-9]+)x([0-9]+)z([0-9]+).", model_path.stem)
        model_name = parse[1]
        h, w = int(parse[2]), int(parse[3])
        downscale = int(parse[4])

        model_class = globals()[model_name]
        K.clear_session()
        model = model_class(input_size=(h, w), weights=model_path)

        for i, img_path in enumerate(image_list):
            if verbose:
                print("{}/{} {}".format(i + 1, len(image_list), img_path.stem))
            img = read_img(img_path, (h, w), downscale=downscale)
            batch = make_tta(img)
            batch = norm_fun(batch)
            if K.image_data_format() == "channels_first":
                batch = np.moveaxis(batch, -1, 1)
            preds = model.predict(batch, batch_size=BATCH_SZ)

            predictions[img_path.stem].extend(preds.flatten())

    return {k: np.array(v) for k, v in predictions.items()}


if __name__ == "__main__":

    output_path = "submission/validation.csv"
    image_list = sorted((ROOT / "validation").glob("*.tif"),
                        key=lambda s: int(s.stem.split("_")[0]) * 1000 + int(s.stem.split("_")[1]))
    predictions = predict(image_list, verbose=True)
    save_predictions(predictions, output_path, std=False, do_rescale=True)
    metrics(ROOT / "val_labels.csv", output_path)

# -----------------------------------------------------------------------------------------------
# Big validation

    # PATIENT_SLIDES = ROOT / "patient_ids.csv"
    # image_list = sorted((ROOT / "train").glob("*.tif"), key=lambda s: int(s.stem.split("_")[1]))
    # patient_df = pd.read_csv(PATIENT_SLIDES)
    #
    # patient_list = patient_df["patient_id"].unique()
    # train_test_split = int(len(patient_list) * 0.2)
    # train_patients = patient_list[train_test_split:]
    # test_patients = patient_list[:train_test_split]
    # train_slides = patient_df.loc[patient_df["patient_id"].isin(train_patients), "slide"].values
    # test_slides = patient_df.loc[patient_df["patient_id"].isin(test_patients), "slide"].values
    #
    # image_list = [img for img in image_list if int(img.stem.split("_")[0]) in test_slides]
    #
    # for i, m in enumerate(MODELS_):
    #     MODELS = [m]
    #     predictions = predict(image_list, verbose=True)
    #     output_path = "submission/bigval/model{}_set2.csv".format(i + 5)
    #     save_predictions(predictions, output_path, std=True, do_rescale=False)
    #     metrics(ROOT / "train_labels.csv", output_path)
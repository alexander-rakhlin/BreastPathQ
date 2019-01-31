# -*- coding: utf-8 -*-

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from datagen import ImageIteratorRegr, train_transform_regr, val_transform_regr
from models import uResNet34regr, uResNet34m46regr, uResNet34Xceptionregr
from pathlib import Path
from model_utils import get_scheduler
import pandas as pd

MODEL_CLASS = uResNet34regr
SIZE = (256, 256)
DOWNSCALE = 2

BATCH_SZ = 4
NB_EPOCH = 600

DATA_ROOT = Path("data/datasets")
LABELS = DATA_ROOT / "train_labels.csv"
PATIENT_SLIDES = DATA_ROOT / "patient_ids.csv"

UNET_WEIGHTS = None  # r"models/uResNet34.sz256x256j0.15z2.100-0.590.set1.hdf5"
MODEL_WEIGHTS = r"models/uResNet34regr.sz256x256z2.64-0.002.semi.S1.hdf5"
MODEL_CHECKPOINT = r"dumps/{}.sz{}x{}z{}.{{epoch:02d}}-{{val_mean_squared_error:.3f}}.semi.S1.set1.hdf5". \
    format(MODEL_CLASS.__name__, *SIZE, DOWNSCALE)
DUMP_FILE = r"dumps/{}.sz{}x{}z{}.semi.S1.hdf5".format(MODEL_CLASS.__name__, *SIZE, DOWNSCALE)
FREEZE_UNET = False

# simple scheduler
LR_STEPS = {0: 5e-4, 3: 1e-4, 10: 5e-5}
# LR_STEPS = {0: 2e-5, 10: 1e-5}
scheduler = get_scheduler(LR_STEPS)

image_list = sorted((DATA_ROOT / "train").glob("*.tif"), key=lambda s: int(s.stem.split("_")[1]))
patient_df = pd.read_csv(PATIENT_SLIDES)

patient_list = sorted(patient_df["patient_id"].unique())
train_test_split = int(len(patient_list) * 0.75)
train_patients = patient_list[:train_test_split]
test_patients = patient_list[train_test_split:]
train_slides = patient_df.loc[patient_df["patient_id"].isin(train_patients), "slide"].values
test_slides = patient_df.loc[patient_df["patient_id"].isin(test_patients), "slide"].values

train_list = [img for img in image_list if int(img.stem.split("_")[0]) in train_slides]
validation_list = [img for img in image_list if int(img.stem.split("_")[0]) in test_slides]

# ---------------------------------------------------------------
# Semisupervised part
# LABELS = "submission/final/test.csv"
# train_list = sorted((DATA_ROOT / "test_patches").glob("*.tif"), key=lambda s: int(s.stem.split("_")[1]))
# LABELS_DF = pd.read_csv(LABELS)
# valid_std = LABELS_DF[LABELS_DF["std"] <= 0.11]
# valid_std = valid_std[["slide", "rid"]].applymap(str).apply(lambda x: "_".join(x), axis=1).values
# train_list = [s for s in train_list if s.stem in valid_std]


if __name__ == "__main__":

    model = MODEL_CLASS(input_size=SIZE, unet_weights=UNET_WEIGHTS, weights=MODEL_WEIGHTS, freeze_unet=FREEZE_UNET)

    train_iterator = ImageIteratorRegr(train_list, LABELS, SIZE,
                                       transform_fun=train_transform_regr(SIZE, downscale=DOWNSCALE),
                                       batch_size=BATCH_SZ, shuffle=True)
    val_iterator = ImageIteratorRegr(validation_list, LABELS, SIZE,
                                     transform_fun=val_transform_regr(SIZE, downscale=DOWNSCALE),
                                     batch_size=BATCH_SZ, shuffle=False)
    callbacks = [LearningRateScheduler(scheduler),
                 ModelCheckpoint(MODEL_CHECKPOINT, monitor="val_mean_squared_error", save_best_only=True)]

    # val_iterator = None
    # callbacks = [LearningRateScheduler(scheduler),
    #              ModelCheckpoint(MODEL_CHECKPOINT, monitor="mean_squared_error", save_best_only=True)]

    model.fit_generator(
        train_iterator,
        steps_per_epoch=len(train_list) // BATCH_SZ,
        epochs=NB_EPOCH,
        validation_data=val_iterator,
        validation_steps=len(validation_list) // BATCH_SZ,
        workers=3,
        callbacks=callbacks
    )
    # model.save(DUMP_FILE)

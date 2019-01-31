# -*- coding: utf-8 -*-

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from datagen import ImageIterator, train_transform, val_transform
from models import uResNet34FPN, uResNet34, u46
from pathlib import Path
from model_utils import get_scheduler

MODEL_CLASS = uResNet34
# SIZE = (448, 448)
SIZE = (256, 256)   # h x w
DOWNSCALE = 2

BATCH_SZ = 4
NB_EPOCH = 1200

JACCARD_WEIGHT = 0.15
THRESHOLD = 0.5
CLASS_WEIGHTS = (1, 1, 4, 0)

ROOT_DIR = Path("")
DATA_ROOT = Path("data/datasets")
MODEL_WEIGHTS = str(ROOT_DIR / r"dumps/uResNet34.sz256x256j0.15z2.100-0.590.set1.hdf5")
MODEL_CHECKPOINT = str(ROOT_DIR / r"dumps/{}.sz{}x{}j{}z{}.{{epoch:02d}}-{{val_jaccard:.3f}}.set1.hdf5".
                       format(MODEL_CLASS.__name__, *SIZE, JACCARD_WEIGHT, DOWNSCALE))

# simple scheduler
LR_STEPS = {0: 5e-4, 10: 1e-4, 40: 5e-5, 60: 2e-5}
# LR_STEPS = {0: 1e-4, 10: 5e-5, 20: 2e-5, 200: 1e-5}

scheduler = get_scheduler(LR_STEPS)

image_list = sorted((DATA_ROOT / "cells").glob("*_crop.tif"), key=lambda s: int(s.name.split("_")[0]))
train_test_split = int(len(image_list) * 0.75)
train_list = image_list[:train_test_split]
validation_list = image_list[train_test_split:]

if __name__ == "__main__":

    model = MODEL_CLASS(input_size=SIZE, weights=MODEL_WEIGHTS, class_weights=CLASS_WEIGHTS,
                        jaccard_weight=JACCARD_WEIGHT, threshold=THRESHOLD)

    train_iterator = ImageIterator(train_list, SIZE, transform_fun=train_transform(SIZE, downscale=DOWNSCALE),
                                   batch_size=BATCH_SZ, shuffle=True)
    val_iterator = ImageIterator(validation_list, SIZE, transform_fun=val_transform(SIZE, downscale=DOWNSCALE),
                                 batch_size=BATCH_SZ, shuffle=False)

    callbacks = [LearningRateScheduler(scheduler), ModelCheckpoint(MODEL_CHECKPOINT, monitor="val_jaccard",
                                                                   save_best_only=True)]

    model.fit_generator(
        train_iterator,
        steps_per_epoch=len(train_list) // BATCH_SZ,
        epochs=NB_EPOCH,
        validation_data=val_iterator,
        validation_steps=len(validation_list) // BATCH_SZ,
        workers=3,
        callbacks=callbacks
    )


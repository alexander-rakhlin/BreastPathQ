# -*- coding: utf-8 -*-

from keras import backend as K

from datagen import ImageIteratorRegr, ImageIterator
from datagen import inference_transform_regr, train_transform_regr, val_transform_regr, train_transform
from models import uResNet34
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from datagen import renorm_fun
from skimage.feature import blob_log
import re
import pickle
from itertools import islice
from threaded_generator import threaded_generator
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor

N_THREADS = 4
NUM_CACHED = 15

SUM_THRESHOLDS = [0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.5]
BLOB_THRESHOLDS = [0.02, 0.04, 0.08, 0.16, 0.24, 0.5]

MODEL_ROOT = Path("models")
DATA_ROOT = Path("data/datasets")
FEATURES_ROOT = Path("data/features")
PATIENT_SLIDES = DATA_ROOT / "patient_ids.csv"


image_list = sorted((DATA_ROOT / "train").glob("*.tif"), key=lambda s: int(s.stem.split("_")[1]))
patient_df = pd.read_csv(PATIENT_SLIDES)

patient_list = sorted(patient_df["patient_id"].unique())
train_test_split = int(len(patient_list) * 0.2)
train_patients = patient_list[train_test_split:]
test_patients = patient_list[:train_test_split]
train_slides = patient_df.loc[patient_df["patient_id"].isin(train_patients), "slide"].values
test_slides = patient_df.loc[patient_df["patient_id"].isin(test_patients), "slide"].values

train_list = [img for img in image_list if int(img.stem.split("_")[0]) in train_slides]
validation_list = [img for img in image_list if int(img.stem.split("_")[0]) in test_slides]
real_validation_list = sorted((DATA_ROOT / "validation").glob("*.tif"),
                              key=lambda s: int(s.stem.split("_")[0]) * 1000 + int(s.stem.split("_")[1]))
real_test_list = sorted((DATA_ROOT / "test_patches").glob("*.tif"),
                              key=lambda s: int(s.stem.split("_")[0]) * 1000 + int(s.stem.split("_")[1]))

N_EPOCHS = 20
BATCH_SZ = 4
SHUFFLE = True
SLIDE_LIST = real_test_list
TRANSFORM_FUN = train_transform_regr
SIZE = (256, 256)
MODEL_WEIGHTS = MODEL_ROOT / "uResNet34Xceptionregr.sz256x256z2.155-0.017.semi.set1.hdf5"
FEATURE_FILE = FEATURES_ROOT / (MODEL_WEIGHTS.stem + ".test" + ".pkl")
LABELS = DATA_ROOT / "train_labels.csv"
LABELS = None


def extract_features(activation):
    features = []
    if K.image_data_format() == "channels_last":
        activation = np.moveaxis(activation, -1, 0)
    for layer in activation[:3]:
        features.append(layer.sum())
        for threshold in SUM_THRESHOLDS:
            bin_mask = layer > threshold
            val_mask = layer[bin_mask]
            features.extend((bin_mask.sum(), val_mask.sum()))
        for threshold in BLOB_THRESHOLDS:
            blobs = blob_log(layer, threshold=threshold, min_sigma=1, max_sigma=4, num_sigma=4, overlap=1)
            blob_count = len(blobs)
            blob_sum = sum(layer[int(y_coord), int(x_coord)] for y_coord, x_coord, _ in blobs)
            features.extend((blob_sum, blob_count))
    return np.array(features, dtype=float)


if __name__ == "__main__":
    MODELS = [
        # MODEL_ROOT / "uResNet34regr.sz256x256z2.109-0.016.semi.S1.set1.hdf5",
        # MODEL_ROOT / "uResNet34regr.sz256x256z2.192-0.017.semi.S2.set1.hdf5",
        MODEL_ROOT / "uResNet34regr.sz256x256z2.113-0.016.semi.S1.set2.hdf5",
        MODEL_ROOT / "uResNet34regr.sz256x256z2.107-0.018.semi.S2.set2.hdf5",
        ]
    for run, MODEL_WEIGHTS in enumerate(MODELS):
        FEATURE_FILE = FEATURES_ROOT / (MODEL_WEIGHTS.stem + ".test" + ".pkl")

        print("Model", MODEL_WEIGHTS.stem)
        parse = re.match("(.+).sz([0-9]+)x([0-9]+)z([0-9]+).", MODEL_WEIGHTS.stem)
        model_name = parse[1].replace("regr", "").replace("Xception", "")
        downscale = int(parse[4])

        model_class = globals()[model_name]
        K.clear_session()
        model = model_class(input_size=SIZE, weights=MODEL_WEIGHTS)

        batch_generator = ImageIteratorRegr(SLIDE_LIST, LABELS, SIZE,
                                            transform_fun=TRANSFORM_FUN(SIZE, downscale=downscale),
                                            batch_size=BATCH_SZ, shuffle=SHUFFLE, output_ids=True)
        if LABELS:
            data = {s.stem: {"x": [], "y": None} for s in SLIDE_LIST}
        else:
            data = {s.stem: {"x": []} for s in SLIDE_LIST}
        total_batches = int(np.ceil(len(SLIDE_LIST) / BATCH_SZ)) * N_EPOCHS

        qu = Queue(maxsize=NUM_CACHED)
        sentinel = object()
        lock = threading.Lock()
        batch_generator_threaded = threaded_generator(batch_generator, num_cached=NUM_CACHED)

        def feature_producer(batch_gen, b_num, model):
            for i, (x_batch, y_batch, ids_batch) in enumerate(islice(batch_gen, b_num)):
                print("Batch {}/{}".format(i + 1, b_num))
                activations = model.predict(x_batch, batch_size=8)
                item = activations, y_batch, ids_batch
                qu.put(item)
            qu.put(sentinel)

        def feature_consumer():
            item = qu.get()
            while item is not sentinel:
                for activation, y, ids in zip(*item):
                    features = extract_features(activation)
                    with lock:
                        data[ids.stem]["x"].append(features)
                        if LABELS:
                            if data[ids.stem]["y"] is not None:
                                assert data[ids.stem]["y"] == y
                            else:
                                data[ids.stem]["y"] = y
                qu.task_done()
                item = qu.get()
                print("Queue size", qu.qsize())
            qu.put(sentinel)    # tell other threads we are done

        with ThreadPoolExecutor() as tpe:
            futures = [tpe.submit(feature_consumer) for _ in range(N_THREADS)]
            feature_producer(batch_generator_threaded, total_batches, model)

        with open(FEATURE_FILE, "wb") as f:
            pickle.dump(data, f)

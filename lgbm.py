import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prediction_probability import predprob
from utils import save_predictions


MODE = "predict"  # predict|train

# FLT = slice(54, None)
FLT = slice(None)
METRIC = "mse"
N_AUG = 20
LEARNING_RATE = 0.01
NUM_ROUND = 6000
MODEL_PATH = Path("models")
TRAIN_FEATURES = Path("data/features/uResNet34regr.sz256x256z2.107-0.018.semi.S2.set2.train.pkl")
VAL_FEATURES = TRAIN_FEATURES.parent / TRAIN_FEATURES.name.replace(".train.", ".val.")
TEST_FEATURES = TRAIN_FEATURES.parent / TRAIN_FEATURES.name.replace(".train.", ".test.")
FOLD = None
MODEL_PATH = MODEL_PATH / TRAIN_FEATURES.name.replace(".train.pkl", ".txt")
PRED_PATH = Path("submission") / TRAIN_FEATURES.name.replace(".train.pkl", ".csv")

PARAM = {
    "objective": METRIC,
    "metric": [METRIC],
    "verbose": 0,
    "learning_rate": LEARNING_RATE,
    "num_leaves": 8,
    "feature_fraction": 0.95,
    "bagging_fraction": 0.5,
    "bagging_freq": 0,
    "max_depth": 5,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
}


def train_test_split(fold=1):
    assert fold in [1, 2]
    ROOT = Path("data/datasets")
    patient_df = pd.read_csv(ROOT / "patient_ids.csv")

    patient_list = patient_df["patient_id"].unique()
    if fold == 1:
        split_ = int(len(patient_list) * 0.75)
        train_patients = patient_list[:split_]
        test_patients = patient_list[split_:]
    elif fold == 2:
        split_ = int(len(patient_list) * 0.2)
        train_patients = patient_list[split_:]
        test_patients = patient_list[:split_]
    train_slides = patient_df.loc[patient_df["patient_id"].isin(train_patients), "slide"].values
    test_slides = patient_df.loc[patient_df["patient_id"].isin(test_patients), "slide"].values
    return train_slides, test_slides


def load_data(p, flt=None, fold=None):
    if flt is None:
        flt = slice(None)
    with open(p, "rb") as f:
        data = pickle.load(f)

    xd, yd, sd = [], [], []
    for slide, v in data.items():
        xx = np.array(v["x"])[:, flt]
        xd.extend(xx)
        sd.extend([slide] * len(xx))
        if "y" in v:
            yd.extend([v["y"][0]] * len(xx))
    xd = np.array(xd)
    yd = np.array(yd) if len(yd) > 0 else None

    if fold is None:
        return xd, yd, sd, None, None, None
    else:
        train_slides, _ = train_test_split(fold)
        idx_train = np.array([int(s.split("_")[0]) in train_slides for s in sd])
        x_train, slides_train = xd[idx_train], [sd[i] for i, l in enumerate(idx_train) if l]
        y_train = None if yd is None else yd[idx_train]
        x_val, slides_val = xd[~idx_train], [sd[i] for i, l in enumerate(idx_train) if ~l]
        y_val = None if yd is None else yd[~idx_train]
    return x_train, y_train, slides_train, x_val, y_val, slides_val


def train(x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array, n_aug=1, save_path=None):

    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    gbm = lgb.train(PARAM, train_data, NUM_ROUND, valid_sets=[train_data, val_data], verbose_eval=True)
    pred = gbm.predict(x_val)

    y_val_reshaped, pred_reshaped, _ = reshape(y_val, pred, n_aug)
    evaluate(y_val_reshaped, pred_reshaped)

    if save_path:
        print("Saving model to", save_path)
        gbm.save_model(str(save_path))

        # control
        print("Loading model to predict...")
        bst = lgb.Booster(model_file=str(save_path))
        # can only predict with the best iteration (or the saving iteration)
        pred = bst.predict(x_val)
        y_val_reshaped, pred_reshaped, _ = reshape(y_val, pred, n_aug)
        print("Control:")
        evaluate(y_val_reshaped, pred_reshaped)


def evaluate(true, pred):
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    pprob = predprob(true, pred)
    print("MSE:", mse, "\nMAE:", mae, "\nPPROB:", pprob)


def reshape(y_true, y_pred, n_aug, slides=None):
    y_pred_reshaped = np.reshape(y_pred, (-1, n_aug))
    y_true_reshaped = None
    if y_true is not None:
        y_true_reshaped = np.reshape(y_true, (-1, n_aug))
        assert np.all(y_true_reshaped == y_true_reshaped[:, [0]])
        y_true_reshaped = y_true_reshaped[:, 0]
    slides_reshaped = None
    if slides is not None:
        slides_reshaped = np.array(slides).reshape(-1, n_aug)
        assert np.all(slides_reshaped == slides_reshaped[:, [0]])
        slides_reshaped = slides_reshaped[:, 0]
    return y_true_reshaped, y_pred_reshaped, slides_reshaped


if __name__ == "__main__":

    if MODE == "train":
        x_train, y_train, _, x_val, y_val, _ = load_data(TRAIN_FEATURES, flt=FLT, fold=FOLD)
        x_val, y_val, _, _, _, _ = load_data(VAL_FEATURES, flt=FLT, fold=None)

        x_train = np.concatenate([x_train, x_val])
        y_train = np.concatenate([y_train, y_val])

        print("Train shape", x_train.shape)
        print("Validation shape", x_val.shape)

        train(x_train, y_train, x_val, y_val, n_aug=N_AUG, save_path=MODEL_PATH)

    elif MODE == "predict":
        x_test, y_test, slides_test, _, _, _ = load_data(TEST_FEATURES, flt=FLT, fold=None)
        print("Predict shape", x_test.shape)

        print("Loading model from", MODEL_PATH)
        booster = lgb.Booster(model_file=str(MODEL_PATH))
        pred = booster.predict(x_test)

        y_test_reshaped, pred_reshaped, slides_test_reshaped = reshape(y_test, pred, N_AUG, slides=slides_test)
        if y_test_reshaped is not None:
            evaluate(y_test_reshaped, pred_reshaped.mean(axis=1))

        # save
        if PRED_PATH:
            dd = dict(zip(slides_test_reshaped, pred_reshaped))
            save_predictions(dd, PRED_PATH, std=True, do_rescale=False)


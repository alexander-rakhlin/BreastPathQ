import pandas as pd
from prediction_probability import predprob
from sklearn.metrics import mean_squared_error
import numpy as np
from pathlib import Path


def metrics(true_path, pred_path, do_rescale=False):
    true_df = pd.read_csv(true_path, index_col=["slide", "rid"]).sort_index()
    if isinstance(pred_path, (str, Path)):
        pred_df = pd.read_csv(pred_path, index_col=["slide", "rid"]).sort_index()
    else:
        pred_df = pred_path

    p_col = "y" if "y" in true_df.columns else "p"
    mse = mean_squared_error(true_df.loc[pred_df.index, p_col], pred_df["p"])
    pprob = predprob(true_df.loc[pred_df.index, p_col], pred_df["p"])
    print("mse={:0.3f}, predprob={:0.3f}".format(mse, pprob))

    if do_rescale:
        pred_df = rescale(pred_df)
        mse = mean_squared_error(true_df.loc[pred_df.index, p_col], pred_df["p"])
        pprob = predprob(true_df.loc[pred_df.index, p_col], pred_df["p"])
        print("mse={:0.3f}, predprob={:0.3f}".format(mse, pprob))


def rescale(df):
    high, low = 0.999, 0.001
    if (df["p"].max() - df["p"].min()) > (high - low):
        df["p"] -= df["p"].min()
        df["p"] /= df["p"].max()
        df["p"] *= high - low
        df["p"] += low
    elif df["p"].min() < low:
        df["p"] -= df["p"].min() - low
    elif df["p"].max() > high:
        df["p"] -= df["p"].max() - high
    return df


def average(*inputs, output):
    from functools import reduce
    input_dfs = [pd.read_csv(i, index_col=["slide", "rid"])["p"].sort_index() for i in inputs]
    output_df = (reduce(pd.Series.add, input_dfs) / len(input_dfs)).reset_index().sort_values(by=["slide", "rid"])
    columns = ["slide", "rid", "p"]
    output_df[columns].to_csv(output, index=False, float_format="%.3f")


def save_predictions(predictions, output_path, std=False, do_rescale=False):
    pr = {k: (v.mean(), v.std()) for k, v in predictions.items()}
    df = pd.DataFrame.from_dict(pr, orient="index", columns=["p", "std"]).reset_index()
    df = pd.concat([df["index"].str.split("_", expand=True), df], axis=1)
    df = df.rename(columns={0: "slide", 1: "rid"})
    df = df.astype({"slide": int, "rid": int}).sort_values(by=["slide", "rid"])

    if do_rescale:
        df = rescale(df)
    if std:
        columns = ["slide", "rid", "p", "std"]
    else:
        columns = ["slide", "rid", "p"]
    df[columns].to_csv(output_path, index=False, float_format="%.3f")


def eval_std():
    pred_df = pd.read_csv("submission/train_full.csv", index_col=["slide", "rid"]).sort_index()
    for std in np.arange(0.02, 0.16, 0.01):
        idx = pred_df["std"] <= std
        print("std {:0.2f}, len {:0.0f}".format(std, sum(idx) / len(idx) * 100), end=" ")
        metrics("data/datasets/train_labels.csv", pred_df[idx])


def final_align(input_path: str):
    pred_df = pd.read_csv(input_path, index_col=["slide", "rid"])
    primer_df = pd.read_csv("data/submission/Sample_Submission_Updated.csv", index_col=["slide", "rid"])

    pred_df = rescale(pred_df)
    pred_df = pred_df.loc[primer_df.index]
    assert not any(pred_df.isnull().any())

    output_path = Path(input_path).parent / "rakhlin_Results.csv"
    columns = ["slide", "rid", "p"]
    pred_df.reset_index()[columns].to_csv(output_path, index=False, float_format="%.3f")
    metrics(input_path, output_path)


if __name__ == "__main__":
    # average(
            # "submission/final/3/lgbm_ensemble.csv",
            # "submission/final/2/test2.csv",
            # output="submission/final/3/lgbm_test2_ensemble.csv")

    metrics("submission/final/1/test1.csv", "submission/final/2/rakhlin_Results.csv")

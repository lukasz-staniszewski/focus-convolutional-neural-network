import argparse
import os
import os.path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from copy import deepcopy

NEW_SHAPE = (640, 360)


def create_csv_balanced(csv_in, csv_out, csv_size):
    df_in = pd.read_csv(csv_in)
    cl_cnts = dict(df_in["category"].value_counts())

    weights = np.array(list(cl_cnts.values()))
    weights = 1 / weights
    weights = weights / weights.sum()

    zip_prob = zip(cl_cnts.keys(), weights)
    scaled_weights = dict(zip_prob)

    for ind, cat in enumerate(list(scaled_weights.keys())):
        cat_df = df_in.loc[df_in["category"] == cat].sample(
            int(scaled_weights[cat] * csv_size), replace=True
        )
        if ind == 0:
            df_out = deepcopy(cat_df)
        else:
            df_out = df_out.append(deepcopy(cat_df))

    df_out.reset_index(drop=True, inplace=True)
    df_out.to_csv(csv_out)


def resize_imgs(inp_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for file in tqdm(os.listdir(inp_dir), desc="Resizing data:"):
        f_img_in = inp_dir + "/" + file
        f_img_out = out_dir + "/" + file
        img = Image.open(f_img_in)
        img = img.resize(NEW_SHAPE)
        img.save(f_img_out)


def main(args):
    arg = args.parse_args()
    if arg.input_dir is None or arg.output_dir is None:
        print("Skipping image resize!")
    else:
        print("~~Resizinig images:~~")
        resize_imgs(arg.input_dir, arg.output_dir)
    if (
        arg.input_csv is None
        or arg.output_csv is None
        or arg.size_csv is None
    ):
        print("Skipping creating balanced csv!")
    print("~~Creating more balanced csv~~")
    create_csv_balanced(
        csv_in=arg.input_csv,
        csv_out=arg.output_csv,
        csv_size=arg.size_csv,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Preprocessor")
    args.add_argument(
        "-i",
        "--input_dir",
        default=None,
        type=str,
        help="Dir of images",
    )
    args.add_argument(
        "-o",
        "--output_dir",
        default=None,
        type=str,
        help="path to output dir",
    )
    args.add_argument(
        "-i_csv",
        "--input_csv",
        default=None,
        type=str,
        help="path to input csv",
    )
    args.add_argument(
        "-o_csv",
        "--output_csv",
        default=None,
        type=str,
        help="path to output csv resampled",
    )
    args.add_argument(
        "-sz_csv",
        "--size_csv",
        default=None,
        type=int,
        help="number of elements in output csv",
    )
    main(args)

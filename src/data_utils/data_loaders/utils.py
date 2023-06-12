from copy import deepcopy
from pathlib import Path
from typing import Union

import pandas as pd


def label_oversample(
    csv_path_train_orig: Union[str, Path], column_label: str = "label"
) -> Union[str, Path]:
    """Function balances training data using oversampling
    minority classes.
    """
    df_orig = pd.read_csv(csv_path_train_orig)
    label_cnt = df_orig[column_label].value_counts().sort_index()
    aug_labels_cnt = label_cnt.max() / label_cnt
    aug_labels_cnt = aug_labels_cnt / aug_labels_cnt.max()
    aug_labels_cnt = (
        aug_labels_cnt[aug_labels_cnt > 0.1] * label_cnt.min()
    ).astype(int)

    dfs_aug = []
    for label, cnt in aug_labels_cnt.iteritems():
        dfs_aug.append(
            df_orig[df_orig[column_label] == label].sample(
                cnt, replace=False, random_state=0
            )
        )
    df_aug = pd.concat(dfs_aug, ignore_index=True)
    csv_path_train_aug = Path(
        str(csv_path_train_orig).replace(".csv", "_aug.csv")
    )

    df_aug.to_csv(csv_path_train_aug, index=False, header=True)
    return csv_path_train_aug


def label_undersample(
    csv_path_train_orig: Union[str, Path],
    balance_max_multiplicity: int,
    csv_path_train_aug: Union[str, Path] = None,
    column_label: str = "label",
) -> Union[str, Path]:
    """Function balances training data using undersampling
    majority classes.

    NOTE: max(x,1) is used to avoid division by zero
    """
    df_orig = pd.read_csv(csv_path_train_orig, header=0)
    new_label_cnt = df_orig[column_label].value_counts().sort_index()
    if csv_path_train_aug:
        df_aug = pd.read_csv(csv_path_train_aug, header=0)
        new_label_cnt = (
            df_orig[column_label]
            .value_counts()
            .add(df_aug[column_label].value_counts(), fill_value=0)
        )

    if (
        new_label_cnt.max() / max(new_label_cnt.min(), 1)
        > balance_max_multiplicity
    ):
        df_undersampled = deepcopy(df_orig)
        min_cnt = new_label_cnt.min()
        for label, cnt in new_label_cnt.iteritems():
            if cnt / max(min_cnt, 1) > balance_max_multiplicity:
                df_undersampled = df_undersampled.drop(
                    df_undersampled[df_undersampled[column_label] == label]
                    .sample(
                        int(cnt - (balance_max_multiplicity * min_cnt)),
                        replace=False,
                        random_state=0,
                    )
                    .index
                )
        csv_path_train_undersampled = Path(
            str(csv_path_train_orig).replace(".csv", "_undersampled.csv")
        )

        df_undersampled.to_csv(
            csv_path_train_undersampled, index=False, header=True
        )
        return csv_path_train_undersampled
    else:
        return csv_path_train_orig


def label_make_0_half(
    csv_path_train_orig: Union[str, Path],
    csv_path_train_aug: Union[str, Path] = None,
    column_label: str = "label",
) -> Union[str, Path]:
    """Function balances training data using undersampling majority classes."""
    df_orig = pd.read_csv(csv_path_train_orig, header=0)
    new_label_cnt = df_orig[column_label].value_counts().sort_index()

    if csv_path_train_aug:
        df_aug = pd.read_csv(csv_path_train_aug, header=0)
        new_label_cnt = (
            df_orig[column_label]
            .value_counts()
            .add(df_aug[column_label].value_counts(), fill_value=0)
        )

    sum_no_0 = sum(new_label_cnt[1:])
    sum_all = sum(new_label_cnt)
    cnt_0 = sum_all - sum_no_0

    if sum_no_0 < cnt_0:
        cnt_to_remove_0 = cnt_0 - sum_no_0
        df_undersampled = deepcopy(df_orig)
        df_undersampled = df_undersampled.drop(
            df_undersampled[df_undersampled[column_label] == 0]
            .sample(int(cnt_to_remove_0), replace=False, random_state=0)
            .index
        )

        csv_path_train_undersampled = Path(
            str(csv_path_train_orig).replace(".csv", "_undersampled.csv")
        )

        df_undersampled.to_csv(
            csv_path_train_undersampled, index=False, header=True
        )
        return csv_path_train_undersampled
    else:
        return csv_path_train_orig

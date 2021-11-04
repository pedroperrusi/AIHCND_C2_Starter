"""
Handy functions for data pre-processing
"""
import os
import glob
import random
import itertools
import numpy as np
import pandas as pd

import sklearn.model_selection


def load_nih_dataset(path_to_data='/data/'):
    """
        Load the NIH data to a pandas dataframe
    :param path_to_data: path to the root of the dataset
    :return:
    """
    xray_df = pd.read_csv(path_to_data + '/Data_Entry_2017.csv')
    all_image_paths = {os.path.basename(x): x for x in
                       glob.glob(os.path.join(path_to_data, 'images*', '*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', xray_df.shape[0])
    xray_df['path'] = xray_df['Image Index'].map(all_image_paths.get)
    return xray_df


def split_findings(df_src):
    """
    Split the 'Finding Labels' column into binary labels

    :param df_src: original dataframe with all findings listed in 'Finding Labels' key, separated by '|'
    :return: Extended dataframe with a key for each label
    """
    d = df_src.copy()
    all_labels = np.unique(list(itertools.chain(*d['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]
    for c_label in all_labels:
        if len(c_label) > 1:  # leave out empty labels
            d[c_label] = d['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
    return d, all_labels


def create_splits(complete_df, validation_size=0.4, train_positive_ratio=0.5, validation_positive_ratio=0.25):
    """
    split your original dataframe into two sets that can be used for training and testing your model
    The proportion of positive labels must be balanced between each dataset.
    This proportion will be achieved by discarding negative samples.

    :param complete_df: Source dataframe with all entries
    :param validation_size: proportion between training and validation sets
    :param validation_positive_ratio: ratio between positive and negative labels in validation set
    :param train_positive_ratio: ratio between positive and negative labels in validation set
    :return: train_df : training dataframe; valid_df: validation dataframe
    """

    train_df, valid_df = sklearn.model_selection.train_test_split(complete_df,
                                                                  test_size=validation_size,
                                                                  stratify=complete_df['Pneumonia'])

    def impose_positive_ratio(df, positive_ratio):
        """ Imposes the required positive ratio by sampling negative samples """
        negative2positive = round(1 / positive_ratio) - 1  # if ratio is not fractional, get the closest
        p_indices = df[df.Pneumonia == 1].index.tolist()
        np_indices = df[df.Pneumonia == 0].index.tolist()
        np_sample = random.sample(np_indices, negative2positive * len(p_indices))
        return df.loc[p_indices + np_sample]

    # Training set will have train_positive_ratio of positives and false labels
    train_df = impose_positive_ratio(train_df, train_positive_ratio)

    # validation set will have a validation_positive_ratio ratio of pneumonia and no pneumonia in the validation set
    valid_df = impose_positive_ratio(valid_df, validation_positive_ratio)

    return train_df, valid_df

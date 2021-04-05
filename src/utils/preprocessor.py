"""
This code is based on these codebases associated with Yuta Saito's research.
- Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback: https://github.com/usaito/unbiased-implicit-rec-real
- Unbiased Pairwise Learning from Biased Implicit Feedback: https://github.com/usaito/unbiased-pairwise-rec
- Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback: https://github.com/usaito/asymmetric-tri-rec-real
"""
import codecs
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


def transform_rating(ratings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Transform ratings into graded relevance information"""
    ratings -= 1
    return eps + (1. - eps) * (2. ** ratings - 1) / (2. ** np.max(ratings) - 1)


def preprocess_datasets(data: str, seed: int = 0) -> Tuple:
    """Load and preprocess raw datasets (Yahoo! R3 or Coat)"""
    if data == 'yahoo':
        with codecs.open(f'../data/{data}/train.txt', 'r', 'utf-8', errors='ignore') as f:
            train_ = pd.read_csv(f, delimiter='\t', header=None)
            train_.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
        with codecs.open(f'../data/{data}/test.txt', 'r', 'utf-8', errors='ignore') as f:
            test_ = pd.read_csv(f, delimiter='\t', header=None)
            test_.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
        for _data in [train_, test_]:
            _data.user, _data.item = _data.user - 1, _data.item - 1
    elif data == 'coat':
        col = {'level_0': 'user', 'level_1': 'item', 2: 'rate', 0: 'rate'}
        with codecs.open(f'../data/{data}/train.ascii', 'r', 'utf-8', errors='ignore') as f:
            train_ = pd.read_csv(f, delimiter=' ', header=None)
            train_ = train_.stack().reset_index().rename(columns=col)
            train_ = train_[train_.rate.values != 0].reset_index(drop=True)
        with codecs.open(f'../data/{data}/test.ascii', 'r', 'utf-8', errors='ignore') as f:
            test_ = pd.read_csv(f, delimiter=' ', header=None)
            test_ = test_.stack().reset_index().rename(columns=col)
            test_ = test_[test_.rate.values != 0].reset_index(drop=True)

    ## Convert rating dataset into implicit feedback one
    # Count the No. of users and items
    num_users, num_items = train_.user.max() + 1, train_.item.max() + 1
    train, test = train_.values, test_.values
    # Transform rating into (0,1)-scale
    test[:, 2] = transform_rating(ratings=test[:, 2], eps=0.0)
    rel_train = np.random.binomial(n=1, p=transform_rating(ratings=train[:, 2], eps=0.1))

    # Extract only positive (relevant) user-item pairs
    train = train[rel_train == 1, :2]
    # Create training data
    all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index().values[:, :2]
    # Note:
    # "train" object includes records which appeared in implicit feedback dataset
    # By subtracting "train" from "all_data", all of "not clicked" records are extracted
    unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    # Impute 0 for unobserved user-item interaction
    train = np.r_[np.c_[train, np.ones(train.shape[0])], np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]
    # Train-validation split using the raw training datasets
    train, val = train_test_split(train, test_size=0.1, random_state=12345)

    return train, val, test, num_users, num_items

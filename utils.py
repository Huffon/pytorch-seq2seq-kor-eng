import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from torchtext import data as ttd
from torchtext.data import Example
from torchtext.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(mode):
    """
    Load train and test dataset and split train dataset to make validation dataset.
    And finally convert train, validation and test dataset to pandas DataFrame.
        mode: (string) configuration mode used to which dataset to load
    Args:

    Returns:
        (DataFrame) train, valid, test dataset converted to pandas DataFrame
    """
    print(f'Loading AI Hub Kor-Eng translation dataset and convert it to pandas DataFrame . . .')

    data_dir = Path().cwd() / 'data'

    if mode == 'train':
        train_file = os.path.join(data_dir, 'train.csv')
        train_data = pd.read_csv(train_file, encoding='utf-8')

        valid_file = os.path.join(data_dir, 'valid.csv')
        valid_data = pd.read_csv(valid_file, encoding='utf-8')

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')

        return train_data, valid_data

    else:
        test_file = os.path.join(data_dir, 'test.csv')
        test_data = pd.read_csv(test_file, encoding='utf-8')

        print(f'Number of testing examples: {len(test_data)}')

        return test_data


def convert_to_dataset(data, kor, eng):
    """
    Pre-process input DataFrame and convert pandas DataFrame to torchtext Dataset.
    Args:
        data: (DataFrame) pandas DataFrame to be converted into torchtext Dataset
        kor: torchtext Field containing Korean sentence
        eng: torchtext Field containing English sentence

    Returns:
        (Dataset) torchtext Dataset
    """
    # drop missing values from DataFrame
    missing_rows = []
    for idx, row in data.iterrows():
        if type(row.korean) != str or type(row.english) != str:
            missing_rows.append(idx)
    data = data.drop(missing_rows)

    # convert each row of DataFrame to torchtext 'Example' which contains kor and eng attributes
    list_of_examples = [Example.fromlist(row.tolist(),
                                         fields=[('kor', kor), ('eng', eng)]) for _, row in data.iterrows()]

    list_of_examples = list_of_examples[:10000]

    # construct torchtext 'Dataset' using torchtext 'Example' list
    dataset = Dataset(examples=list_of_examples, fields=[('kor', kor), ('eng', eng)])

    return dataset


def make_iter(batch_size, mode, train_data=None, valid_data=None, test_data=None):
    """
    Convert pandas DataFrame to torchtext Dataset and make iterator used to train and test
    Args:
        batch_size: (integer) batch size used to make iterators
        mode: (string) configuration mode used to which iterator to make
        train_data: (DataFrame) pandas DataFrame used to make train iterator
        valid_data: (DataFrame) pandas DataFrame used to make validation iterator
        test_data: (DataFrame) pandas DataFrame used to make test iterator

    Returns:
        (BucketIterator) train, valid, test iterator
    """
    # load text and label field made by build_pickles.py
    file_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(file_kor)
    pad_idx = kor.vocab.stoi[kor.pad_token]

    file_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(file_eng)

    # convert pandas DataFrame to torchtext dataset
    if mode == 'train':
        train_data = convert_to_dataset(train_data, kor, eng)
        valid_data = convert_to_dataset(valid_data, kor, eng)

        # make iterator using train and validation dataset
        print(f'Make Iterators for training . . .')
        train_iter, valid_iter = ttd.BucketIterator.splits(
            (train_data, valid_data),
            # the BucketIterator needs to be told what function it should use to group the data.
            # In our case, we sort dataset using text of example
            sort_key=lambda sent: len(sent.kor),
            # all of the tensors will be sorted by their length by below option
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return train_iter, valid_iter, pad_idx

    else:
        test_data = convert_to_dataset(test_data, kor, eng)
        dummy = list()

        # make iterator using test dataset
        print(f'Make Iterators for testing . . .')
        test_iter, _ = ttd.BucketIterator.splits(
            (test_data, dummy),
            sort_key=lambda sent: len(sent.kor),
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return test_iter, pad_idx


def init_weights(model):
    """
    Seq2Seq paper introduces the method to initialize the model's parameter.
    And this method implements that methodology
    :param model: Model object whose parameters will be initialized with the value between -0.08 and 0.08
    :return:
    """
    for _, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def epoch_time(start_time, end_time):
    """
    Calculate the time spent to train one epoch
    Args:
        start_time: (float) training start time
        end_time: (float) training end time

    Returns:
        (int) elapsed_mins and elapsed_sec spent for one epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

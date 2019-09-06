import os
import pickle
import argparse
import pandas as pd

from pathlib import Path
from utils import convert_to_dataset

from torchtext import data as ttd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer


def build_tokenizer():
    """
    Train soynlp tokenizer which will be used to tokenize input sentence
    Returns:

    """
    print(f'Now building soy-nlp tokenizer . . .')
    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv')

    df = pd.read_csv(train_file, encoding='utf-8')

    kor_lines = []
    # if encounters non-text row, skips it
    for idx, row in df.iterrows():
        if type(row.korean) != str or type(row.english) != str:
            continue
        kor_lines.append(row.korean)

    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(kor_lines)

    word_scores = word_extractor.extract()
    cohesion_scores = {word: score.cohesion_forward for word, score in word_scores.items()}

    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)


def build_vocab(config):
    """
    Build vocab used to convert input sentence into word indices using soynlp tokenizer
    Args:
        config: configuration containing various options
        tokenizer: soynlp tokenizer used to struct torchtext Field object

    Returns:

    """

    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    # To use packed padded sequences, tell the model how long the actual sequences are
    kor = ttd.Field(tokenize=tokenizer.tokenize,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

    eng = ttd.Field(tokenize='spacy',
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_file, encoding='utf-8')
    train_data = convert_to_dataset(train_data, kor, eng)

    print(f'Build vocabulary using torchtext . . .')
    kor.build_vocab(train_data, min_freq=2)
    eng.build_vocab(train_data, min_freq=2)

    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}')
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}')

    print(f'Most commonly used Korean words are as follows:')
    print(kor.vocab.freqs.most_common(20))

    print(f'Most commonly used English words are as follows:')
    print(eng.vocab.freqs.most_common(20))

    file_kor = open('pickles/kor.pickle', 'wb')
    pickle.dump(kor, file_kor)

    file_eng = open('pickles/eng.pickle', 'wb')
    pickle.dump(eng, file_eng)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build pickles used to use model')

    parser.add_argument('--kor_vocab', type=int, default=50000)
    parser.add_argument('--eng_vocab', type=int, default=25000)

    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)

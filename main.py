import pickle
import argparse
import torch

from trainer import Trainer
from utils import load_dataset, make_iter


def main(config):
    if config.mode == 'train':
        train_data, valid_data = load_dataset(config.mode)
        train_iter, valid_iter, pad_idx = make_iter(config.batch_size, config.mode, train_data=train_data,
                                                    valid_data=valid_data)

        trainer = Trainer(config, pad_idx, train_iter=train_iter, valid_iter=valid_iter)

        trainer.train()

    else:
        test_data = load_dataset(config.mode)
        test_iter, pad_idx = make_iter(config.batch_size, config.mode, test_data=test_data)
        trainer = Trainer(config, pad_idx, test_iter=test_iter)

        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence to sequence')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=32)
    parser.add_argument('--clip', type=int, default=1)

    # Model Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--bidirectional', type=bool, default=True)

    # Additional options
    parser.add_argument('--model', type=str, default='seq2seq', choices=['seq2seq'])
    parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--save_model', type=str, default='model.pt')

    config = parser.parse_args()

    # load kor and eng vocabs to add vocab size configuration
    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)
    config.kor_vocab_size = len(kor.vocab)

    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)
    config.eng_vocab_size = len(eng.vocab)

    # add device information to the configuration object
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(config)

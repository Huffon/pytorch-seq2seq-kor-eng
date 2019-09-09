import argparse

from trainer import Trainer
from utils import load_dataset, make_iter, Params


def main(config):
    params_dict = {
        'seq2seq': Params('configs/params.json'),
        'seq2seq_gru': Params('configs/params_gru.json'),
        'seq2seq_attention': Params('configs/params_attention.json')
    }

    params = params_dict[config.model]

    if config.mode == 'train':
        train_data, valid_data = load_dataset(params.mode)
        train_iter, valid_iter = make_iter(params.batch_size, params.mode, train_data=train_data, valid_data=valid_data)

        trainer = Trainer(params, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()

    else:
        params.mode = "test"

        test_data = load_dataset(params.mode)
        test_iter = make_iter(params.batch_size, params.mode, test_data=test_data)

        trainer = Trainer(params, test_iter=test_iter)
        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence to sequence')

    # Additional options
    parser.add_argument('--model', type=str, default='seq2seq', choices=['seq2seq', 'seq2seq_gru', 'seq2seq_attention'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    main(config)

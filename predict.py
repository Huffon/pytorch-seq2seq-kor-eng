import pickle
import argparse
import torch

from utils import Params
from soynlp.tokenizer import LTokenizer

from models.seq2seq import Seq2Seq
from models.seq2seq_gru import Seq2SeqGRU
from models.seq2seq_attention import Seq2SeqAttention


def predict(params):
    params_dict = {
        'seq2seq': Params('configs/params.json'),
        'seq2seq_gru': Params('configs/params_gru.json'),
        'seq2seq_attention': Params('configs/params_attention.json'),
    }

    params = params_dict[params.model]

    # load tokenizer and torchtext Fields
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)

    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)

    model_type = {
        'seq2seq': Seq2Seq,
        'seq2seq_gru': Seq2SeqGRU,
        'seq2seq_attention': Seq2SeqAttention,
    }

    # select model and load trained model
    model = model_type[params.model](params)
    model.load_state_dict(torch.load(params.save_model))
    model.eval()

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(config.input)
    tokenized = ['<sos>'] + tokenized + ['<eos>']
    indexed = [kor.vocab.stoi[token] for token in tokenized]

    tensor = torch.LongTensor(indexed).to(params.device)  # [input length]
    tensor = tensor.unsqueeze(1)  # [input length, 1] : unsqueeze(1) to add batch size dimension

    translation_tensor_logits = model(tensor, None, 0)
    # translation_tensor_logits = [target length, 1, eng_vocab_size]

    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    # translation_tensor = [target length]

    translation = [eng.vocab.itos[token] for token in translation_tensor]
    translation = ' '.join(translation[1:])

    print(f'"{config.input}" is translated into "{translation}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kor-Eng Translation prediction')

    # Additional options
    parser.add_argument('--model', type=str, default='seq2seq', choices=['seq2seq', 'seq2seq_gru'])
    parser.add_argument('--input', type=str, default='이 문서는 제출 할 필요 없어요.')

    config = parser.parse_args()

    predict(config)

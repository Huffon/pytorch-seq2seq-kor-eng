import pickle
import argparse
import torch

from soynlp.tokenizer import LTokenizer
from models.seq2seq import Seq2Seq


def predict(config):
    # load tokenizer and torchtext Fields

    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)

    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)

    model_type = {
        'seq2seq': Seq2Seq(config),
    }

    # select model and load trained model
    model = model_type[config.model]
    model.load_state_dict(torch.load(config.save_model))
    model.eval()

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(config.input)
    tokenized = ['<sos>'] + tokenized + ['<eos>']
    indexed = [kor.vocab.stoi[token] for token in tokenized]

    tensor = torch.LongTensor(indexed).to(config.device)  # [input length]
    tensor = tensor.unsqueeze(1)  # [input length, 1] : unsqueeze(1) to add batch size dimension

    translation_tensor_logits = model(tensor, None, 0)
    # translation_tensor_logits = [target length, 1, eng_vocab_size]

    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    # translation_tensor = [target length]

    translation = [eng.vocab.itos[token] for token in translation_tensor]

    print(f'{config.input} is translated into {translation[1:]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kor-Eng Translation prediction')

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
    parser.add_argument('--save_model', type=str, default='model.pt')
    parser.add_argument('--input', type=str, default='이 문서는 제출 할 필요 없어요.')

    config = parser.parse_args()

    # load kor and eng vocabs to add vocab size configuration
    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)
    config.kor_vocab_size = len(kor.vocab)

    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)
    config.eng_vocab_size = len(eng.vocab)

    # add device information to configuration
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # add <sos> and <eos> tokens' indices used to predict the target sentence
    config.sos_idx = eng.vocab.stoi['<sos>']
    config.eos_idx = eng.vocab.stoi['<eos>']

    predict(config)

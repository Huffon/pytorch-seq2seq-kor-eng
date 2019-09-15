import re
import pickle
import argparse

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from soynlp.tokenizer import LTokenizer

from utils import Params
from models.seq2seq import Seq2Seq
from models.seq2seq_gru import Seq2SeqGRU
from models.seq2seq_attention import Seq2SeqAttention


def clean_text(text):
    """
    remove special characters from the input sentence to normalize it
    Args:
        text: (string) text string which may contain special character

    Returns:
        normalized sentence
    """
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text)
    return text


def predict(config):
    params_dict = {
        'seq2seq': Params('configs/params.json'),
        'seq2seq_gru': Params('configs/params_gru.json'),
        'seq2seq_attention': Params('configs/params_attention.json'),
    }

    params = params_dict[config.model]

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
    model = model_type[config.model](params)

    model.load_state_dict(torch.load(params.save_model))
    model.to(params.device)
    model.eval()

    input = clean_text(config.input)

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(input)
    indexed = [kor.vocab.stoi[token] for token in tokenized]

    source_length = torch.LongTensor([len(indexed)]).to(params.device)

    tensor = torch.LongTensor(indexed).unsqueeze(1).to(params.device)  # [source length, 1]: unsqueeze to add batch size

    if config.model == 'seq2seq_attention':
        translation_tensor_logits, attention = model(tensor, source_length, None, 0)
        # translation_tensor_logits = [target length, 1, output dim]

        translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
        # translation_tensor = [target length] filed with word indices

        translation = [eng.vocab.itos[token] for token in translation_tensor][1:]
        attention = attention[1:]

        display_attention(tokenized, translation, attention)
    else:
        translation_tensor_logits = model(tensor, source_length, None, 0)
        translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
        translation = [eng.vocab.itos[token] for token in translation_tensor][1:]

    translation = ' '.join(translation)
    print(f'kor> {config.input}')
    print(f'eng> {translation.capitalize()}')


def display_attention(candidate, translation, attention):
    """
    displays the model's attention over the source sentence for each target token generated.
    Args:
        candidate: (list) tokenized source tokens
        translation: (list) predicted target translation tokens
        attention: a tensor containing attentions scores

    Returns:
    """
    # attention = [target length, batch size (1), source length]

    attention = attention.squeeze(1).cpu().detach().numpy()
    # attention = [target length, source length]

    font_location = 'pickles/NanumSquareR.ttf'
    fontprop = fm.FontProperties(fname=font_location)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + [t.lower() for t in candidate], rotation=45, fontproperties=fontprop)
    ax.set_yticklabels([''] + translation, fontproperties=fontprop)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kor-Eng Translation prediction')

    # Vocabulary size options
    parser.add_argument('--model', type=str, default='seq2seq', choices=['seq2seq', 'seq2seq_gru', 'seq2seq_attention'])
    parser.add_argument('--input', type=str, default='오늘 우리는 맛있는 저녁을 먹을거야')

    config = parser.parse_args()

    predict(config)

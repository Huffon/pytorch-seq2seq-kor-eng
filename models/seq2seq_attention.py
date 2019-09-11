import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(params.input_dim, params.embed_dim)

        # the dropout is used between each layer of a multi-layered RNN
        # as we only use a single layer, don't pass the dropout as an argument
        self.gru = nn.GRU(params.embed_dim,
                          params.enc_hidden_dim,
                          bidirectional=True)

        self.fc = nn.Linear(params.enc_hidden_dim * 2, params.dec_hidden_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, input):
        # input = [input length, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [input length, batch size, embed dim]

        output, hidden = self.gru(embedded)
        # output = [input length, batch size, enc hidden dim * num directions]
        # hidden = [num layers * num directions, batch size, enc hidden dim]
        # [forward_1, backward_1, forward_2, backward_2, ..., forward_n, backward_n]

        # since the decoder is not bidirectional, it needs a single context vector to use as its initial hidden state,
        # so concatenate the two context vectors, pass it through a linear layer and apply the tanh activation function
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # hidden = [batch size, dec hidden dim]

        # as we want our model to look back over the whole of the source sentence return outputs,
        # the stacked forward and backward hidden states for every token in the source sentence
        # also return hidden, which acts as initial hidden state in the decoder
        return output, hidden


class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()

        self.attention = nn.Linear((params.enc_hidden_dim * 2) + params.dec_hidden_dim, params.dec_hidden_dim)
        self.v = nn.Parameter(torch.rand(params.dec_hidden_dim))

    def forward(self, hidden, encoder_output):
        # takes previous hidden state from the decoder and stacked hidden states from the encoder
        # hidden = [batch size, hidden dim]
        # encoder_output = [input length, batch size, enc hidden dim * num directions]

        hidden = hidden.unsqueeze(1)
        # hidden = [batch size, 1, dec hidden dim]

        batch_size = encoder_output.shape[1]
        source_len = encoder_output.shape[0]

        # repeat one-size hidden state 'input length' times to make it has same size as the encoder output
        hidden = hidden.repeat(1, source_len, 1)
        encoder_output = encoder_output.permute(1, 0, 2)
        # hidden = [batch size, input length, dec hidden dim]
        # encoder_output = [batch size, input length, enc hidden dim * num directions]

        # calculate how each encoder hidden state "matches" the previous decoder hidden state
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_output), dim=2)))
        # energy = [batch size, input length, dec hidden dim]

        energy = energy.permute(0, 2, 1)
        # energy = [batch size, dec hidden dim, input length]

        # convert energy to be [batch size, input length] as the attention should be over the input length
        # this is achieved by multiplying the energy by a [batch size, 1, dec hidden dim] tensor

        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch size, 1, dec hidden dim]

        # calculate a weighted sum of the "match" over all dec hidden dim elements for each encoder hidden state,
        # where the weights are learned (as we learn the parameters of v)

        # bmm is a batch matrix-matrix product: [batch size, a, b] * [batch size, b, c] = [batch size, a, c]
        attention = torch.bmm(v, energy).squeeze(1)
        # attention = [batch size, input length]

        # returns attention vector with input length, each element is between 0 and 1 and the entire vector sums to 1
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.attention = Attention(params)
        self.embedding = nn.Embedding(params.output_dim, params.embed_dim)

        self.gru = nn.GRU((params.enc_hidden_dim * 2) + params.embed_dim,
                          params.dec_hidden_dim)

        self.fc = nn.Linear(params.embed_dim + params.dec_hidden_dim + (params.enc_hidden_dim * 2), params.output_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, input, hidden, encoder_output):
        # input  = [batch size]
        # hidden = [batch size, dec hidden dim]
        # encoder_output = [input length, batch size, enc hidden dim * num directions]
        # re-use the same context vector returned by the encoder for every time-step in the decoder

        # add additional dimension to make the shape of the input same as the input dimension of Embedding layer
        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embed dim]

        # takes the previous hidden state, all of the encoder hidden states and returns the attention vector
        attention = self.attention(hidden, encoder_output)
        # attention = [batch size, input length]

        attention = attention.unsqueeze(1)
        # attention = [batch size, 1, input length]

        encoder_output = encoder_output.permute(1, 0, 2)
        # encoder_output = [batch size, input length, enc hidden dim * num directions]

        # create a weighted input vector which is a weighted sum of the encoder hidden states using attention

        # [batch size, 1, input length] * [batch size, input length, enc hidden dim * num directions]
        # = [batch size, 1, enc hidden dim * num directions]
        weighted = torch.bmm(attention, encoder_output)
        # weighted = [batch size, 1, enc hidden dim * num directions]

        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hidden dim * num directions]

        gru_input = torch.cat((embedded, weighted), dim=2)
        # gru_input = [1, batch size, embed dim +(enc hidden dim * num directions)]

        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        # output = [1, batch size, dec hidden dim]
        # hidden = [1, batch size, dec hidden dim]

        assert (output == hidden).all()

        output = output.squeeze(0)      # [batch size, dec hidden dim]: current hidden state
        embedded = embedded.squeeze(0)  # [batch size, embed dim]: current target token
        weighted = weighted.squeeze(0)  # [batch size, enc hidden dim * num directions]: weighted vector using attention

        output = self.fc(torch.cat((output, weighted, embedded), dim=1))
        # output = [batch size, output dim]

        # return predicted output and hidden state which will be used to initialize the next decoder
        return output, hidden.squeeze(0)


class Seq2SeqAttention(nn.Module):
    def __init__(self, params):
        super(Seq2SeqAttention, self).__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target, teacher_forcing=0.5):
        # source = [source length, batch size]
        # target = [target length, batch size]

        # if target is None, check whether teacher_forcing is zero
        if target is None:
            assert teacher_forcing == 0, "Must be zero during inference"

            # makes inference flag True and defines dummy target sentences with max length as 100
            inference = True

            target = torch.zeros((100, source.shape[1])).long().fill_(self.params.sos_idx).to(self.params.device)
            # target = [100, 1] filled with <sos> tokens
        else:
            inference = False

        # the length of the output shouldn't exceeds the lengths of target sentences
        target_max_len = target.shape[0]
        # batch size changes dynamically, so updates the batch size each time step
        batch_size = target.shape[1]

        # define 'outputs' tensor used to store each time step's output ot the decoder
        outputs = torch.zeros(target_max_len, batch_size, self.params.output_dim).to(self.params.device)
        # outputs = [target length, batch size, eng_vocab_size]

        # encoder_output is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_output, hidden = self.encoder(source)

        # initial input to the decoder is <sos> tokens, naming 'output' to use generically
        output = target[0, :]
        # output = [batch size]

        for time in range(1, target_max_len):
            output, hidden = self.decoder(output, hidden, encoder_output)
            # output contains the predicted results which has the size of output dim (eng_vocab_size)
            # output = [batch size, output dim]

            # store the output of each time step to the 'outputs' tensor
            outputs[time] = output

            # calculates boolean flag whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing

            # pick the token which has the largest value from each time step's output
            # output.max(1) is a tuple containing (the largest tokens's probability, index of that token)
            top1 = output.max(1)[1]
            # top1 = [batch size]

            # if use teacher forcing, next input token is the ground-truth token
            # if we don't, next input token is the predicted token. naming 'output' to use generically
            output = (target[time] if teacher_force else top1)

            # during inference time, when encounters <eos> token, return generated outputs
            if inference and output.item() == self.params.eos_idx:
                return outputs[:time]

        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

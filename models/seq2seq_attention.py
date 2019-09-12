import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(params.input_dim, params.embed_dim)

        self.gru = nn.GRU(params.embed_dim,
                          params.enc_hidden_dim,
                          bidirectional=True)

        self.fc = nn.Linear(params.enc_hidden_dim * 2, params.dec_hidden_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, source, source_length):
        # source = [source length, batch size]

        embedded = self.dropout(self.embedding(source))
        # embedded = [source length, batch size, embed dim]

        packed_embedded = pack_padded_sequence(embedded, source_length)
        packed_output, hidden = self.gru(packed_embedded)

        output, _ = pad_packed_sequence(packed_output)   # pad tokens are all-zeros
        # output = [source length, batch size, enc hidden dim * 2]
        # hidden = [num layers * 2, batch size, enc hidden dim]
        # [forward_1, backward_1, forward_2, backward_2, ..., forward_n, backward_n]

        # since the decoder is not bidirectional, it needs a 'single' context vector to initialize its hidden state,
        # so concatenate the two context vectors, pass it through a linear layer and apply the tanh activation function
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # hidden = [batch size, dec hidden dim]

        # as we want our model to look back over the whole of the source sentence, return output with final hidden
        return output, hidden


class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()

        self.attention = nn.Linear(params.dec_hidden_dim + (params.enc_hidden_dim * 2), params.dec_hidden_dim)
        self.v = nn.Parameter(torch.rand(params.dec_hidden_dim))

    def forward(self, hidden, encoder_output, mask):
        # takes previous hidden state from the decoder and stacked hidden states from the encoder
        # hidden         = [batch size, dec hidden dim]
        # encoder_output = [source length, batch size, enc hidden dim * 2]
        # mask           = [batch size, source length]
        # mask consists of 1 when the source sentence token is not a padding token, and 0 when it is a padding token

        hidden = hidden.unsqueeze(1)
        # hidden = [batch size, 1, dec hidden dim]

        # those will be used to 'repeat'
        batch_size = encoder_output.shape[1]
        source_len = encoder_output.shape[0]

        # repeat one-size hidden state 'source length' times to make it has same size as the encoder output
        hidden = hidden.repeat(1, source_len, 1)
        encoder_output = encoder_output.permute(1, 0, 2)
        # hidden         = [batch size, source length, dec hidden dim]
        # encoder_output = [batch size, source length, enc hidden dim * 2]

        # calculate how each encoder hidden state "matches" the previous decoder hidden state
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_output), dim=2)))
        # energy = [batch size, source length, dec hidden dim]

        energy = energy.permute(0, 2, 1)
        # energy = [batch size, dec hidden dim, source length]

        # convert energy to be [batch size, source length] as the attention should be over the source length
        # this is achieved by multiplying the energy by a [batch size, 1, dec hidden dim] tensor

        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch size, 1, dec hidden dim]

        # calculate a weighted sum of the "match" over all 'dec hidden dim' elements,
        # where the weights are learned (as we learn the parameters of v) # parameterized attention

        # bmm is a batch matrix-matrix product: [batch size, a, b] * [batch size, b, c] = [batch size, a, c]
        attention = torch.bmm(v, energy).squeeze(1)
        # attention = [batch size, source length]

        # using masking, force the attention to only be over non-padding elements
        attention = attention.masked_fill(mask == 0, -1e10)

        # returns attention vector with source length, each element is between 0 and 1 and the entire vector sums to 1
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.attention = Attention(params)
        self.embedding = nn.Embedding(params.output_dim, params.embed_dim)

        self.gru = nn.GRU(params.embed_dim + (params.enc_hidden_dim * 2),
                          params.dec_hidden_dim)

        self.fc = nn.Linear(params.dec_hidden_dim + params.embed_dim + (params.enc_hidden_dim * 2), params.output_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, target, hidden, encoder_output, mask):
        # target         = [batch size]
        # hidden         = [batch size, dec hidden dim]
        # encoder_output = [source length, batch size, enc hidden dim * 2]
        # mask           = [batch size, source length]
        # re-use the same context vector returned by the encoder for every time-step in the decoder

        target = target.unsqueeze(0)
        # target = [1, batch size]

        embedded = self.dropout(self.embedding(target))
        # embedded = [1, batch size, embed dim]

        # takes the previous hidden state, all of the encoder hidden states and returns the attention vector
        attention = self.attention(hidden, encoder_output, mask)
        # attention = [batch size, source length]

        attention = attention.unsqueeze(1)
        # attention = [batch size, 1, source length]

        encoder_output = encoder_output.permute(1, 0, 2)
        # encoder_output = [batch size, source length, enc hidden dim * 2]

        # create a weighted context vector which is a weighted sum of the encoder hidden states using attention
        #   [batch size, 1, source length] * [batch size, source length, enc hidden dim * 2]
        # = [batch size, 1, enc hidden dim * 2]
        weighted = torch.bmm(attention, encoder_output)
        # weighted = [batch size, 1, enc hidden dim * 2]

        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hidden dim * 2]

        gru_input = torch.cat((embedded, weighted), dim=2)
        # gru_input = [1, batch size, embed dim + (enc hidden dim * 2)]

        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        # output = [1, batch size, dec hidden dim]
        # hidden = [1, batch size, dec hidden dim]

        assert (output == hidden).all()

        output = output.squeeze(0)      # [batch size, dec hidden dim]    : current decoder hidden state
        embedded = embedded.squeeze(0)  # [batch size, embed dim]         : current input target token
        weighted = weighted.squeeze(0)  # [batch size, enc hidden dim * 2]: weighted vector made by using attention

        output = self.fc(torch.cat((output, weighted, embedded), dim=1))
        # output = [batch size, output dim]

        # return a predicted output, a new hidden state and attention tensor
        return output, hidden.squeeze(0), attention.squeeze(1)


class Seq2SeqAttention(nn.Module):
    def __init__(self, params):
        super(Seq2SeqAttention, self).__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def create_mask(self, source):
        mask = (source != self.params.pad_idx).permute(1, 0)
        # mask = [batch size, source length]
        return mask

    def forward(self, source, source_length, target, teacher_forcing=0.5):
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

        target_max_len = target.shape[0]
        batch_size = target.shape[1]

        outputs = torch.zeros(target_max_len, batch_size, self.params.output_dim).to(self.params.device)
        # outputs = [target length, batch size, output dim]

        # define a tensor to store attentions that show which source tokens 'are looked up' by target tokens
        attentions = torch.zeros(target_max_len, batch_size, source.shape[0]).to(self.params.device)
        # attentions = [target length, batch size, source length]

        # encoder_output is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_output, hidden = self.encoder(source, source_length)

        # initial input to the decoder is <sos> tokens
        input = target[0, :]
        # input = [batch size]

        mask = self.create_mask(source)
        # mask = [batch size, source length]

        for time in range(1, target_max_len):
            output, hidden, attention = self.decoder(input, hidden, encoder_output, mask)
            # output contains the predicted results which has the size of output dim
            # output    = [batch size, output dim]
            # hidden    = [batch size, dec hidden dim]
            # attention = [batch size, source length]

            # store the output and attention of each time step at the 'outputs' and 'attentions' tensor respectively
            outputs[time] = output
            attentions[time] = attention

            # calculates boolean flag whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing

            # pick the token which has the largest value from each time step's output
            # output.max(1) is a tuple containing (the largest tokens's probability, index of that token)
            top1 = output.max(1)[1]
            # top1 = [batch size]

            # if use teacher forcing, next input token is the ground-truth token
            # if we don't, next input token is the predicted token. naming 'output' to use generically
            input = (target[time] if teacher_force else top1)

            # during inference time, when encounters <eos> token, return generated outputs and attentions
            if inference and input.item() == self.params.eos_idx:
                return outputs[:time], attentions[:time]

        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

import numpy as np
import torch

from params import *


class LockedDropout(torch.nn.Module):
    '''
    DOCSTRING
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html
    '''

    def __init__(self, p=0.5):

        super().__init__()

        self.p = p

    def forward(self, inp_pack):

        if not self.training or not self.p:
            return inp_pack

        inp_pad, inp_pad_lens = torch.nn.utils.rnn.pad_packed_sequence(inp_pack, batch_first=True)
        inp_pad = inp_pad.clone()

        mask = inp_pad.new_empty(inp_pad.shape[0], 1, inp_pad.shape[2], requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(inp_pad)

        out_pad = inp_pad * mask
        out_pack = torch.nn.utils.rnn.pack_padded_sequence(out_pad, inp_pad_lens,
                                                           batch_first=True, enforce_sorted=False)

        return out_pack

    def __repr__(self):

        return self.__class__.__name__ + '(' + 'p=' + str(self.p) + ')'


class pBLSTM(torch.nn.Module):
    '''
    DOCSTRING
    TODO: generalize for all factors
    '''

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.blstm = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   bidirectional=True,
                                   batch_first=True)

    def forward(self, inp_pack):

        inp_pad, inp_pad_lens = torch.nn.utils.rnn.pad_packed_sequence(inp_pack, batch_first=True)
        inp_pad_trunc, inp_pad_trunc_lens = self.trunc_reshape(inp_pad, inp_pad_lens)
        inp_pack_trunc = torch.nn.utils.rnn.pack_padded_sequence(inp_pad_trunc, inp_pad_trunc_lens,
                                                                 batch_first=True, enforce_sorted=False)

        out_pack_trunc, _ = self.blstm(inp_pack_trunc)

        return out_pack_trunc

    def trunc_reshape(self, inp_pad, inp_lens):

        batch_size, time_steps, feature_len = inp_pad.shape

        # drop last if odd number of time steps
        inp_pad = inp_pad[:, :(time_steps//2)*2, :]

        # concatenate consecutive time steps along features length
        inp_trunc = inp_pad.reshape(batch_size, time_steps//2, feature_len*2)
        inp_trunc_lens = torch.clamp(inp_lens, max=time_steps//2)

        return inp_trunc, inp_trunc_lens


class Listener(torch.nn.Module):
    '''
    DOCSTRING
    TODO: generalize for all factors
    '''

    def __init__(self, input_size, encoder_hidden_size):

        super().__init__()

        self.base_lstm = torch.nn.LSTM(input_size=input_size,
                                       hidden_size=encoder_hidden_size,
                                       num_layers=1,
                                       bidirectional=True,
                                       batch_first=True)

        self.ld1 = LockedDropout(0.5)
        self.pBLSTM1 = pBLSTM(input_size=encoder_hidden_size*(2*2),
                              hidden_size=encoder_hidden_size)

        self.ld2 = LockedDropout(0.4)
        self.pBLSTM2 = pBLSTM(input_size=encoder_hidden_size*(2*2),
                              hidden_size=encoder_hidden_size)

        self.ld3 = LockedDropout(0.3)
        self.pBLSTM3 = pBLSTM(input_size=encoder_hidden_size*(2*2),
                              hidden_size=encoder_hidden_size)

    def forward(self, inp_pad, inp_pad_lens):

        inp_pack = torch.nn.utils.rnn.pack_padded_sequence(inp_pad, inp_pad_lens,
                                                           batch_first=True, enforce_sorted=False)

        base_lstm_pack, _ = self.base_lstm(inp_pack)
        ld1_pack = self.ld1(base_lstm_pack)

        pblstm1_lstm_pack = self.pBLSTM1(ld1_pack)
        ld2_pack = self.ld2(pblstm1_lstm_pack)

        pblstm2_lstm_pack = self.pBLSTM2(ld2_pack)
        ld3_pack = self.ld3(pblstm2_lstm_pack)

        pblstm3_lstm_pack = self.pBLSTM3(ld3_pack)

        pblstm_pad, pblstm_pad_lens = torch.nn.utils.rnn.pad_packed_sequence(pblstm3_lstm_pack, batch_first=True)

        return pblstm_pad, pblstm_pad_lens


class Attention(torch.nn.Module):
    '''
    DOCSTRING
    TODO: generalize for all factors
    TODO: context projection
        self.context_projection = torch.nn.Linear()

    key   : (batch_size, timesteps, projection_size)
    value : (batch_size, timesteps, projection_size)
    query : (batch_size, projection_size)
    '''

    def __init__(self, encoder_hidden_size, decoder_output_size, projection_size):

        super().__init__()

        self.projection_size = projection_size

        self.key_projection = torch.nn.Linear(2*encoder_hidden_size, projection_size)
        self.value_projection = torch.nn.Linear(2*encoder_hidden_size, projection_size)
        self.query_projection = torch.nn.Linear(decoder_output_size, projection_size)
        self.context_projection = torch.nn.Linear(self.projection_size, self.projection_size)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, decoder_output_embedding):

        self.query = self.query_projection(decoder_output_embedding)

        _, query_len = self.query.shape
        raw_weights = torch.divide(torch.bmm(self.key, torch.unsqueeze(self.query, dim=2)),
                                   np.sqrt(query_len)).squeeze(2)

        masked_raw_weights = raw_weights.masked_fill_(self.padding_mask, float('-inf'))
        attention_weights = self.softmax(masked_raw_weights)

        context = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)
        context = self.context_projection(context)

        return context, attention_weights

    def set_key_value_mask(self, encoder_outputs, encoder_lens):

        _, encoder_max_seq_len, _ = encoder_outputs.shape

        self.key = self.key_projection(encoder_outputs)
        self.value = self.value_projection(encoder_outputs)

        range_tensor = torch.arange(encoder_max_seq_len).unsqueeze(0)
        encoder_lens = encoder_lens.unsqueeze(1)

        self.padding_mask = (range_tensor >= encoder_lens).to(next(self.parameters()).device)


class Speller(torch.nn.Module):
    '''
    DOCSTRING
    '''

    def __init__(self, embed_size, decoder_hidden_size, decoder_output_size, vocab_size, attention_module=None):

        super().__init__()

        self.vocab_size = vocab_size
        self.projection_size = 0

        self.attention = attention_module

        if attention_module is not None:
            self.projection_size = self.attention.projection_size

        self.embedding = torch.nn.Embedding(vocab_size, embed_size, padding_idx=EOS_TOKEN)

        self.lstm_cells = torch.nn.Sequential(torch.nn.LSTMCell(embed_size+self.projection_size, decoder_hidden_size),
                                              torch.nn.LSTMCell(decoder_hidden_size, decoder_output_size))

        self.char_prob = torch.nn.Linear(decoder_output_size+self.projection_size, vocab_size)

        # initialize lstm cell weights
        for _, param in self.lstm_cells.named_parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

        # weight tying
        self.char_prob.weight = self.embedding.weight

    def forward(self, encoder_outputs, encoder_lens, y=None, tf_rate=1):

        batch_size, encoder_max_seq_len, _ = encoder_outputs.shape

        if self.training:
            _, timesteps = y.shape
            label_embed = self.embedding(y)
        else:
            timesteps = 600

        predictions = []
        attention_plot = []

        # initialize the first character input to <sos>
        char = torch.full((batch_size,), fill_value=SOS_TOKEN, dtype=torch.long).to(next(self.parameters()).device)

        # list to keep track of LSTM Cell hidden and memory states
        hidden_states = [None] * len(self.lstm_cells)

        context = torch.zeros((batch_size, self.projection_size)).to(next(self.parameters()).device)
        attention_weights = torch.zeros(batch_size, encoder_max_seq_len)

        if self.attention is not None:
            self.attention.set_key_value_mask(encoder_outputs, encoder_lens)

        for t in range(timesteps):

            char_embed = self.embedding(char)

            if self.training and t > 0 and np.random.rand() < tf_rate:
                char_embed = label_embed[:, t-1, :]

            decoder_input_embedding = torch.cat([char_embed, context], dim=1)

            for i, cell in enumerate(self.lstm_cells):
                hidden_states[i] = cell(decoder_input_embedding, hidden_states[i])
                decoder_input_embedding = hidden_states[i][0]

            decoder_output_embedding = hidden_states[-1][0]

            if self.attention is not None:
                context, attention_weights = self.attention(decoder_output_embedding)

            output_embedding = torch.cat([self.attention.query, context], dim=1)
            char_pred = self.char_prob(output_embedding)

            attention_plot.append(attention_weights[0].detach().cpu())
            predictions.append(char_pred)

            char = torch.argmax(char_pred, dim=1)

        attention_plot = torch.stack(attention_plot, dim=0)
        predictions = torch.stack(predictions, dim=1)

        return predictions, attention_plot


class LAS(torch.nn.Module):
    '''
    DOCSTRING
    '''

    def __init__(self, input_size, encoder_hidden_size,
                 vocab_size, embed_size,
                 decoder_hidden_size, decoder_output_size,
                 projection_size=128):

        super().__init__()

        self.encoder = Listener(input_size, encoder_hidden_size)

        attention_module = Attention(encoder_hidden_size, decoder_output_size, projection_size)

        self.decoder = Speller(embed_size, decoder_hidden_size, decoder_output_size,
                               vocab_size, attention_module)

    def forward(self, x, lx, y=None, tf_rate=0.0):

        encoder_outputs, encoder_lens = self.encoder(x, lx)
        predictions, attention_plot = self.decoder(encoder_outputs, encoder_lens, y, tf_rate)

        return predictions, attention_plot


def get_model():

    model = LAS(input_size=CONFIG['input_size'], encoder_hidden_size=CONFIG['encoder_hidden_size'],
                vocab_size=len(VOCAB), embed_size=CONFIG['embed_size'],
                decoder_hidden_size=CONFIG['decoder_hidden_size'], decoder_output_size=CONFIG['decoder_output_size'],
                projection_size=CONFIG['projection_size'])
    
    return model

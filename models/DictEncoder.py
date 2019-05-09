import torch
import torch.nn as nn
import torch.nn.functional as F

from .EncoderRNN import EncoderRNN

class DictEncoder(nn.Module):
    def __init__(self, vocab_size, pos_size, tag_size, max_len, hidden_size,
            input_dropout_p=0, dropout_p=0,
            pretrained_embedding=None, train_embedding=True,
            n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=False):

        super(DictEncoder, self).__init__()

        self.bidirectional = bidirectional
        self.n_layers = n_layers

        # Encoder RNN
        self.rnn_cell = rnn_cell
        self.encoder = EncoderRNN(vocab_size, max_len, hidden_size,
                           input_dropout_p=input_dropout_p,
                           dropout_p=dropout_p,
                           pretrained_embedding=pretrained_embedding,
                           train_embedding=train_embedding,
                           n_layers=n_layers,
                           bidirectional=bidirectional,
                           rnn_cell=rnn_cell,
                           variable_lengths=variable_lengths)

        # POS Embedding
        self.embed_pos = nn.Embedding(pos_size, hidden_size)
        self.embed_pos.weight.data.normal_(mean=0, std=0.5)

        # TAG Embedding
        self.embed_tag = nn.Embedding(tag_size, hidden_size)
        self.embed_tag.weight.data.normal_(mean=0, std=0.5)

        # Fusion layers
        if self.bidirectional:
            self.fuse_hiddens = nn.Linear(hidden_size * (n_layers * 2), hidden_size)
        else:
            self.fuse_hiddens = nn.Linear(hidden_size * n_layers, hidden_size)


    def forward(self, defn, defn_lengths, pos, tag):
        pos_emb = self.embed_pos(pos)
        tag_emb = self.embed_tag(tag)

        pos_emb = pos_emb.unsqueeze(0)

        if self.bidirectional:
            pos_emb = pos_emb.expand(2 * self.n_layers, -1, -1).contiguous()
        else:
            pos_emb = pos_emb.expand(self.n_layers, -1, -1).contiguous()

        if self.rnn_cell is 'lstm':
            h0 = (pos_emb, pos_emb)
            output, (h, c) = self.encoder(defn, defn_lengths)

        else:
            h0 = pos_emb
            output, h = self.encoder(defn, defn_lengths)

        _, B, D = h.size()
        h = h.transpose(0, 1).contiguous().view(B, -1)
        out = self.fuse_hiddens(h) + tag_emb

        return out

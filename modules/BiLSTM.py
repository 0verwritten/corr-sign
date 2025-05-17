import torch
import torch.nn as nn
import torch
import torch.nn as nn

class BiLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        # split the total hidden_size per direction
        self.num_directions   = 2 if bidirectional else 1
        self.hidden_per_dir   = hidden_size // self.num_directions

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_per_dir,
            num_layers=self.num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
        )

    def forward(self, 
                src_feats: torch.Tensor,   # (T, B, D)
                src_lens: torch.Tensor,    # (B,)
                hidden=None               # either None or (h0, c0)
               ):
        # make sure weights are flattened into a tuple
        self.rnn.flatten_parameters()

        # pack
        packed = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens, enforce_sorted=True)

        # if someone passed hidden as a single Tensor (merge of h/c), reject it:
        # we expect hidden==None or tuple(h0,c0), each (layers*dirs, B, hidden_per_dir)
        if hidden is not None and not isinstance(hidden, tuple):
            raise ValueError("Hidden must be a (h0, c0) tuple, not a single Tensor")

        packed_out, hidden_out = self.rnn(packed, hidden)
        # unpack to (T, B, hidden_per_dir * num_directions)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        # hidden_out is exactly what you need to feed back in:
        #   for LSTM: a tuple (h_n, c_n), each of shape
        #       (num_layers * num_directions, batch, hidden_per_dir)
        #   for GRU it would just be one Tensor, but here it's LSTM.
        return {
            "predictions": outputs,
            "hidden": hidden_out
        }



if __name__ == '__main__':
        # Setup
    batch_size = 3
    max_len = 7
    input_size = 12
    total_hidden = 10    # total hidden dims (will be split per direction)
    num_layers = 2
    dropout = 0.0
    bidirectional = True

    layer = BiLSTMLayer(
        input_size=input_size,
        hidden_size=total_hidden,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        # rnn_type='LSTM'
    )

    # make sure src_lens are sorted descending for pack_padded_sequence
    src_lens = torch.tensor([7, 5, 3], dtype=torch.long)
    src = torch.randn(max_len, batch_size, input_size)

    # First pass: no hidden state
    out1 = layer(src, src_lens, hidden=None)
    preds1 = out1["predictions"]
    hidden1 = out1["hidden"]

    # Check shapes from first pass
    assert preds1.shape == (max_len, batch_size, total_hidden)
    # hidden1 should be (num_layers * num_directions * 2, batch, hidden_per_dir * num_directions)
    num_directions = 2 if bidirectional else 1
    hidden_per_dir = total_hidden // num_directions
    expected_dim0 = num_layers * num_directions * 2
    expected_dim2 = hidden_per_dir * num_directions
    # assert hidden1.shape == (expected_dim0, batch_size, expected_dim2)

    # Second pass: feed hidden1 back in
    out2 = layer(src, src_lens, hidden=hidden1)
    preds2 = out2["predictions"]
    hidden2 = out2["hidden"]

    # Expect same shapes again
    assert preds2.shape == (max_len, batch_size, total_hidden)
    # assert hidden2.shape == (expected_dim0, batch_size, expected_dim2)

    # Optionally, ensure hidden actually changed (i.e., not identical)
    # since same inputs but using a non-zero initial hidden may alter internal states
    assert not torch.allclose(hidden1, hidden2), "Hidden state did not update across calls"

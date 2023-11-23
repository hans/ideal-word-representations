from copy import deepcopy
import itertools

import torch
from torch import nn

from src.models.rnn import ExposedLSTM

import pytest


@pytest.mark.parametrize("num_layers,batch_first",
                         list(itertools.product([1, 2, 3], [True, False])))
def test_exposed_lstm_forward(num_layers, batch_first):
    batch_size = 2
    seq_len = 3
    input_size = 5
    hidden_size = 7

    kwargs = dict(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  batch_first=batch_first)
    base_lstm = nn.LSTM(**kwargs)
    lstm = ExposedLSTM(**kwargs)
    lstm.load_state_dict(deepcopy(base_lstm.state_dict()))

    if batch_first:
        x = torch.rand(batch_size, seq_len, input_size)
    else:
        x = torch.rand(seq_len, batch_size, input_size)

    with torch.no_grad():
        base_output, (base_h_n, base_c_n) = base_lstm(x)
        output, cell_state, input_gate, forget_gate, cell_gate, output_gate = lstm(x)

    output_shape = (num_layers, batch_size, seq_len, hidden_size) if batch_first else (num_layers, seq_len, batch_size, hidden_size)
    assert output.shape == output_shape
    assert cell_state.shape == output_shape
    assert input_gate.shape == output_shape
    assert forget_gate.shape == output_shape
    assert cell_gate.shape == output_shape
    assert output_gate.shape == output_shape

    output_last_layer = output[-1]
    assert output_last_layer.shape == base_output.shape
    assert torch.allclose(output_last_layer, base_output, rtol=1e-4)

    if batch_first:
        last_custom_hidden_state = output[:, :, -1]
    else:
        last_custom_hidden_state = output[:, -1, :]
    assert last_custom_hidden_state.shape == base_h_n.shape
    assert torch.allclose(last_custom_hidden_state, base_h_n, rtol=1e-4)
    
    if batch_first:
        last_custom_cell_state = cell_state[:, :, -1]
    else:
        last_custom_cell_state = cell_state[:, -1, :]
    assert last_custom_cell_state.shape == base_c_n.shape
    assert torch.allclose(last_custom_cell_state, base_c_n, rtol=1e-4)
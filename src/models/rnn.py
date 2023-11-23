
import torch
from torch import nn


class ExposedLSTM(nn.Module):
    """
    Functionally equivalent (for our use case) of torch.nn.LSTM, but exposes the
    intermediate states (gate values) of the LSTM computation.
    """

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True,
                 batch_first=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Define LSTM parameters
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            setattr(self, f"weight_ih_l{i}", nn.Parameter(torch.Tensor(4 * hidden_size, layer_input_size)))
            setattr(self, f"weight_hh_l{i}", nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size)))
            setattr(self, f"bias_ih_l{i}", nn.Parameter(torch.Tensor(4 * hidden_size)))
            setattr(self, f"bias_hh_l{i}", nn.Parameter(torch.Tensor(4 * hidden_size)))

        # Initialize parameters
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, initial_states=None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        all_h_t, all_c_t, all_i_t, all_f_t, all_g_t, all_o_t = [], [], [], [], [], []
        for i in range(self.num_layers):
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device),
                        torch.zeros(self.hidden_size).to(x.device)) if initial_states is None else initial_states

            h_t, c_t = h_t.view(1, -1).expand(batch_size, -1), c_t.view(1, -1).expand(batch_size, -1)

            h_t_list, c_t_list, i_t_list, f_t_list, g_t_list, o_t_list = [], [], [], [], [], []

            W_ii, W_if, W_ic, W_io = torch.split(getattr(self, f"weight_ih_l{i}"), self.hidden_size, dim=0)
            W_hi, W_hf, W_hc, W_ho = torch.split(getattr(self, f"weight_hh_l{i}"), self.hidden_size, dim=0)
            b_ii, b_if, b_ic, b_io = torch.split(getattr(self, f"bias_ih_l{i}"), self.hidden_size, dim=0)
            b_hi, b_hf, b_hc, b_ho = torch.split(getattr(self, f"bias_hh_l{i}"), self.hidden_size, dim=0)

            for t in range(seq_len):
                if i == 0:
                    x_t = x[t]
                else:
                    x_t = all_h_t[-1][t]

                i_t = torch.sigmoid(x_t @ W_ii.t() + b_ii + h_t @ W_hi.t() + b_hi)
                f_t = torch.sigmoid(x_t @ W_if.t() + b_if + h_t @ W_hf.t() + b_hf)
                g_t = torch.tanh(x_t @ W_ic.t() + b_ic + h_t @ W_hc.t() + b_hc)
                o_t = torch.sigmoid(x_t @ W_io.t() + b_io + h_t @ W_ho.t() + b_ho)

                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

                h_t_list.append(h_t)
                c_t_list.append(c_t)
                i_t_list.append(i_t)
                f_t_list.append(f_t)
                g_t_list.append(g_t)
                o_t_list.append(o_t)

            h_t = torch.stack(h_t_list, dim=0)
            c_t = torch.stack(c_t_list, dim=0)
            i_t = torch.stack(i_t_list, dim=0)
            f_t = torch.stack(f_t_list, dim=0)
            g_t = torch.stack(g_t_list, dim=0)
            o_t = torch.stack(o_t_list, dim=0)

            all_h_t.append(h_t)
            all_c_t.append(c_t)
            all_i_t.append(i_t)
            all_f_t.append(f_t)
            all_g_t.append(g_t)
            all_o_t.append(o_t)

        all_h_t = torch.stack(all_h_t, dim=0)
        all_c_t = torch.stack(all_c_t, dim=0)
        all_i_t = torch.stack(all_i_t, dim=0)
        all_f_t = torch.stack(all_f_t, dim=0)
        all_g_t = torch.stack(all_g_t, dim=0)
        all_o_t = torch.stack(all_o_t, dim=0)

        if self.batch_first:
            all_h_t = all_h_t.transpose(1, 2)
            all_c_t = all_c_t.transpose(1, 2)
            all_i_t = all_i_t.transpose(1, 2)
            all_f_t = all_f_t.transpose(1, 2)
            all_g_t = all_g_t.transpose(1, 2)
            all_o_t = all_o_t.transpose(1, 2)

        return all_h_t, all_c_t, all_i_t, all_f_t, all_g_t, all_o_t
        # return h_t, c_t, i_t, f_t, g_t, o_t
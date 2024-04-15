"""
Taken from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
authors: TUM-LMF
"""
import torch.nn as nn
from torch.autograd import Variable
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device):
        return (
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, pad_mask=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        pad_maks (b , t)
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=input_tensor.size(0), device=input_tensor.device
            )

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            if pad_mask is not None:
                last_positions = (~pad_mask).sum(dim=1) - 1
                layer_output = layer_output[:, last_positions, :, :, :]

            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTM_Seg(nn.Module):
    def __init__(
        self, num_classes, input_size, input_dim, hidden_dim, kernel_size, pad_value=0
    ):
        super(ConvLSTM_Seg, self).__init__()
        self.convlstm_encoder = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        self.classification_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=1,
        )
        self.pad_value = pad_value

    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        pad_mask = pad_mask if pad_mask.any() else None
        _, states = self.convlstm_encoder(input, pad_mask=pad_mask)
        out = states[0][1]  # take last cell state as embedding
        out = self.classification_layer(out)

        return out


class BConvLSTM_Seg(nn.Module):
    def __init__(
        self, num_classes, input_size, input_dim, hidden_dim, kernel_size, pad_value=0
    ):
        super(BConvLSTM_Seg, self).__init__()
        self.convlstm_forward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        self.convlstm_backward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        self.classification_layer = nn.Conv2d(
            in_channels=2 * hidden_dim,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=1,
        )
        self.pad_value = pad_value

    def forward(self, input, batch_posistions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        pad_mask = pad_mask if pad_mask.any() else None

        # FORWARD
        _, forward_states = self.convlstm_forward(input, pad_mask=pad_mask)
        out = forward_states[0][1]  # take last cell state as embedding

        # BACKWARD
        x_reverse = torch.flip(input, dims=[1])
        if pad_mask is not None:
            pmr = torch.flip(pad_mask.float(), dims=[1]).bool()
            x_reverse = torch.masked_fill(x_reverse, pmr[:, :, None, None, None], 0)
            # Fill leading padded positions with 0s
        _, backward_states = self.convlstm_backward(x_reverse)

        out = torch.cat([out, backward_states[0][1]], dim=1)
        out = self.classification_layer(out)
        return out


class BConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        super(BConvLSTM, self).__init__()
        self.convlstm_forward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        self.convlstm_backward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )

    def forward(self, input, pad_mask=None):
        # FORWARD
        _, forward_states = self.convlstm_forward(input, pad_mask=pad_mask)
        out = forward_states[0][1]  # take last cell state as embedding

        # BACKWARD
        x_reverse = torch.flip(input, dims=[1])
        if pad_mask is not None:
            pmr = torch.flip(pad_mask.float(), dims=[1]).bool()
            x_reverse = torch.masked_fill(x_reverse, pmr[:, :, None, None, None], 0)
            # Fill leading padded positions with 0s
        _, backward_states = self.convlstm_backward(x_reverse)

        out = torch.cat([out, backward_states[0][1]], dim=1)
        return out
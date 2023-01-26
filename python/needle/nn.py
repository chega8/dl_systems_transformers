"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
from needle.functional import softmax
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.device = device
        self.weight = Parameter(
          init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype
          )
        self.bias = Parameter(
          init.kaiming_uniform(out_features, 1).reshape((1, out_features)), device=device, dtype=dtype
          )if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = X @ self.weight
        if self.bias: res += self.bias.broadcast_to(res.shape)
        return res
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # x: a tensor of shape (B,X_0,X_1,...)
        # return: x flattened to the shape of (B, X_0 * X_1 * ...)
        B = X.shape[0]
        k = np.prod(list(X.shape[1:]))
        return X.reshape((B, k))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (1 + ops.exp(-x)) ** (-1)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # n: batch_num; k: class_num
        n, k, *_ = logits.shape
        y_one_hot = init.one_hot(k, y, device=y.device)
        # axes=(1,)
        logsumexp = ops.logsumexp(logits, axes=(1,))
        z_y = (logits * y_one_hot).sum(axes=(1,))
        return (logsumexp - z_y).sum() / n
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype,requires_grad=True),device=device,dtype=dtype)
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype,requires_grad=True),device=device,dtype=dtype)
        self.running_mean = init.zeros(dim,device=device,dtype=dtype)
        self.running_var = init.ones(dim,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, dim = x.shape
        assert dim == self.dim
        e, var, y = None, None, None
        weight = self.weight.reshape((1, dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, dim)).broadcast_to(x.shape)
        if (self.training):
            e = x.sum(axes=(0,)) / n
            self.running_mean.data = ((1 - self.momentum) * self.running_mean + self.momentum * e).data
            e = e.reshape((1, dim)).broadcast_to(x.shape)
            var = ((x - e) ** 2).sum(axes=(0,)) / n
            self.running_var.data = ((1 - self.momentum) * self.running_var + self.momentum * var).data
            var = var.reshape((1, dim)).broadcast_to(x.shape)
        else:
            e = self.running_mean.reshape((1, dim)).broadcast_to(x.shape).data
            var = self.running_var.reshape((1, dim)).broadcast_to(x.shape).data

        norm = ((x - e) / ((var + self.eps) ** 0.5))
        y = weight * norm + bias
        if (self.training == False):
            y = y.data
        return y
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype),device=device,dtype=dtype)
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype),device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: (n x dim)
        # e: (1 x dim)
        # var:(1 x dim)
        n, dim = x.shape
        assert dim==self.dim
        e = (x.sum(axes=(1,)) / dim).reshape((n,1)).broadcast_to((n,dim))
        var = (((x-e)**2).sum(axes=(1,)) / dim).reshape((n,1)).broadcast_to((n,dim))
        weight = self.weight.reshape((1,dim)).broadcast_to((n,dim))
        bias = self.bias.reshape((1,dim)).broadcast_to((n,dim))
        y = weight * (x-e) / ((var+self.eps)**0.5) + bias
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        if self.training:
          # probi = 0 with probability p
          prob = init.randb(*x.shape, p = 1-self.p)
          y = x/(1-self.p) * prob
        return y
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.use_bias = bias
        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels * kernel_size * kernel_size,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
            ),
            device=device,
            dtype=dtype,
        )

        bound = 1 / math.sqrt(in_channels * kernel_size**2)
        self.bias = Parameter(
            init.rand(out_channels, low=-bound, high=bound), device=device, dtype=dtype
        ) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.conv(
            x.transpose((1, 2)).transpose((2, 3)),
            self.weight,
            self.stride,
            self.kernel_size // 2,
        )
        if self.use_bias:
            out += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(
                (*out.shape[:-1], self.out_channels)
            )
        return out.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = bias
        self.activation = {"tanh": Tanh(), "relu": ReLU()}[nonlinearity]
        bound = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.bias_ih = Parameter(
            init.rand(hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        self.bias_hh = Parameter(
            init.rand(hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        out = X @ self.W_ih + h @ self.W_hh
        if self.has_bias:
            out += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(out.shape)
            out += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(out.shape)
        return self.activation(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias,
                nonlinearity,
                device=device,
                dtype=dtype,
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X_split = ops.split(X, 0)
        if h0 is None:
            h0 = init.zeros(
                self.num_layers,
                X.shape[1],
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
        h_split = list(ops.split(h0, 0))
        out = []
        for i in range(X.shape[0]):
            for j in range(self.num_layers):
                h_split[j] = self.rnn_cells[j](
                    X_split[i] if j == 0 else h_split[j - 1], h_split[j]
                )
            out += [h_split[-1]]
        return ops.stack(out, 0), ops.stack(h_split, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = bias
        bound = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, 4 * hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.bias_ih = Parameter(
            init.rand(4 * hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        self.bias_hh = Parameter(
            init.rand(4 * hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        self.activation = [Sigmoid(), Sigmoid(), Tanh(), Sigmoid()]
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = (
                init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype),
            ) * 2
        h0, c0 = h
        out = ops.reshape(
            X @ self.W_ih + h0 @ self.W_hh, (X.shape[0], 4, self.hidden_size)
        )
        if self.has_bias:
            out += self.bias_ih.reshape((1, *out.shape[1:])).broadcast_to(out.shape)
            out += self.bias_hh.reshape((1, *out.shape[1:])).broadcast_to(out.shape)
        out = ops.split(out, 1)
        i, f, g, o = [x(out[i]) for i, x in enumerate(self.activation)]
        c_next = f * c0 + i * g
        h_next = o * ops.tanh(c_next)
        return h_next, c_next
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias,
                device=device,
                dtype=dtype,
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X_split = ops.split(X, 0)
        if h is None:
            h = (
                init.zeros(
                    self.num_layers,
                    X.shape[1],
                    self.hidden_size,
                    device=X.device,
                    dtype=X.dtype,
                ),
            ) * 2
        h_split, c_split = [list(ops.split(_, 0)) for _ in h]
        out = []
        for i in range(X.shape[0]):
            for j in range(self.num_layers):
                h_split[j], c_split[j] = self.lstm_cells[j](
                    X_split[i] if j == 0 else h_split[j - 1], (h_split[j], c_split[j])
                )
            out += [h_split[-1]]
        return ops.stack(out, 0), (ops.stack(h_split, 0), ops.stack(c_split, 0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.
        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector
        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim), device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors
        Input:
        x of shape (seq_len, bs)
        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, batch_size = x.shape
        x_one_hot = init.one_hot(
            self.num_embeddings,
            x.reshape((seq_len * batch_size,)),
            device=x.device,
            dtype=x.dtype,
        )
        return (x_one_hot @ self.weight).reshape((*x.shape, self.embedding_dim))


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A gated recurrent unit (GRU) cell.
        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights
        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 3*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 3*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (3*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (3*hidden_size,).
        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = bias
        bound = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, 3 * hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, 3 * hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.bias_ih = Parameter(
            init.rand(3 * hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        self.bias_hh = Parameter(
            init.rand(3 * hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        self.sigmoid = Sigmoid()

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.
        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        out_1 = ops.reshape(X @ self.W_ih, (X.shape[0], 3, self.hidden_size))
        out_2 = ops.reshape(h @ self.W_hh, (X.shape[0], 3, self.hidden_size))
        if self.has_bias:
            out_1 += self.bias_ih.reshape((1, *out_1.shape[1:])).broadcast_to(out_1.shape)
            out_2 += self.bias_hh.reshape((1, *out_2.shape[1:])).broadcast_to(out_2.shape)
        out_1 = ops.split(out_1, 1)
        out_2 = ops.split(out_2, 1)
        z = self.sigmoid(out_1[0] + out_2[0])
        r = self.sigmoid(out_1[1] + out_2[1])
        n = ops.tanh(out_1[2] + r * out_2[2])
        h_next = (1 - z) * n + z * h
        return h_next


class GRU(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.
        Variables:
        gru_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 3*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 3*hidden_size).
        gru_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 3*hidden_size).
        gru_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (3*hidden_size,).
        gru_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (3*hidden_size,).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = [
            GRUCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias,
                device=device,
                dtype=dtype,
            )
            for i in range(num_layers)
        ]

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.
        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the GRU, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        X_split = ops.split(X, 0)
        if h0 is None:
            h0 = init.zeros(
                self.num_layers,
                X.shape[1],
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
        h_split = list(ops.split(h0, 0))
        out = []
        for i in range(X.shape[0]):
            for j in range(self.num_layers):
                h_split[j] = self.gru_cells[j](
                    X_split[i] if j == 0 else h_split[j - 1], h_split[j]
                )
            out += [h_split[-1]]
        return ops.stack(out, 0), ops.stack(h_split, 0)
    
    
class MultiHeadedAttention(Module):
    def __init__(self, h, d_model, dropout=0.1, device=None, dtype="float32"):
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % h == 0
        
        self.d_head = d_model // h
        self.h = h
        
        self.attn = None
        self.dropout = Dropout(p=dropout)
        
        self.dtype = dtype
        self.device = device
        # TODO: implement split() function like numpy so that devide weight matrix to K V Q
        # weight - W_KQV matrix of shape (d_model, d_model * 3)
        # self.weight = Parameter(
        #     init.kaiming_uniform(d_model, d_model * 3), device=device, dtype=dtype
        # )
        self.w_k = Parameter(
            init.kaiming_uniform(d_model, d_model), device=device, dtype=dtype
        )
        self.w_q = Parameter(
            init.kaiming_uniform(d_model, d_model), device=device, dtype=dtype
        )
        self.w_v = Parameter(
            init.kaiming_uniform(d_model, d_model), device=device, dtype=dtype
        )
        self.w_out = Parameter(
            init.kaiming_uniform(d_model, d_model), device=device, dtype=dtype
        )
    
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        # TODO: implement split() function like numpy so that devide weight matrix to K V Q
        # key, query, value = ops.split(x @ self.weight, 3, axis=-1)
        key = bmm(x, self.w_k)
        query = bmm(x, self.w_q)
        value = bmm(x, self.w_v)

        key = self.__split_heads(key, batch_size, seq_len, d_model)
        query = self.__split_heads(query, batch_size, seq_len, d_model)
        value = self.__split_heads(value, batch_size, seq_len, d_model)
        
        print(f"key: {key.shape}, query.T: {query.transpose().shape}")
        attn = softmax(key @ query.transpose() / np.sqrt(d_model // self.h) + mask)
        output = (attn @ value).transpose((1,2)).reshape(batch_size, seq_len, d_model)
        output = output @ self.w_out
        return output, attn
    
    def __split_heads(self, key, batch_size, seq_len, d_model):
        new_shape = (batch_size, seq_len, self.h, d_model // self.h)
        return key.reshape(new_shape).transpose((1, 2))
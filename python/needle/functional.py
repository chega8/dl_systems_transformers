import numpy as np

from .autograd import Tensor
from .ops import exp
from .backend_selection import array_api, NDArray


def keepdims(x, old_shape, axis):
    shape = list(old_shape)
    shape[axis] = 1
    return x.reshape(tuple(shape))

def softmax(logits: Tensor):
    norm_logits = logits - logits.maximum(-1).detach().numpy().max(axis=-1, keepdims=True)
    logits = exp(norm_logits)
    ax = len(logits.shape) - 1
    sum_logits = keepdims(logits.sum(axes=(ax)), logits.shape, ax)
    return logits / sum_logits
    
def self_attention(X, mask, W_KQV, W_out):
    K, Q, V = np.split(X @ W_KQV, 3, axis=-1)
    attn = softmax(K @ Q.swapaxes(-1, -2) / np.sqrt(X.shape[-1]) + mask)
    return attn @ V @ W_out, attn
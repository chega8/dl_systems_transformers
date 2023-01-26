import numpy as np

from .autograd import Tensor
from .ops import exp
from functools import reduce
from .backend_selection import array_api, NDArray
import needle as ndl


def keepdims(x, old_shape, axis):
    shape = list(old_shape)
    shape[axis] = 1
    return x.reshape(tuple(shape)).broadcast_to(old_shape)

def softmax(logits: Tensor):
    
    max_norm = logits \
                .detach() \
                .numpy() \
                .max(axis=-1, keepdims=True)
    max_norm = np.broadcast_to(max_norm, logits.shape)
    norm_logits = logits - ndl.Tensor(max_norm, requires_grad=False)
    logits = exp(norm_logits)
    ax = len(logits.shape) - 1
    sum_logits = keepdims(logits.sum(axes=(ax)), logits.shape, ax)
    return logits / sum_logits

def self_attention(X, mask, W_KQV, W_out):
    K, Q, V = np.split(X @ W_KQV, 3, axis=-1)
    attn = softmax(K @ Q.swapaxes(-1, -2) / np.sqrt(X.shape[-1]) + mask)
    return attn @ V @ W_out, attn

# def batch_matmul(batch1, batch2):
#     batch1_shape = batch1.shape
#     batch2_shape = batch2.shape
    
#     seq_len, d_model = batch1_shape[-2:]
#     hiden_dim = batch2_shape[-1]
#     assert d_model == batch2_shape[-2]
    
#     flatten_bs = reduce(lambda a, b: a * b, batch1_shape[:-2])
#     batches1 = split(batch1.reshape((flatten_bs, seq_len, d_model)), axis=0)
        
#     batch = []
#     if batch1.ndim == batch2.ndim:
#         assert batch1_shape[:-2] == batch1_shape[:-2]
#         batches2 = split(batch2.reshape((flatten_bs, d_model, hiden_dim)), axis=0)
        
#         for i in range(flatten_bs):
#             batch.append(batches1[i] @ batches2[i])
        
#     else:
#         print(f"len {len(batches1)}, type {type(batches1)}")
#         print(f"flatten_bs {flatten_bs}")
#         for i in range(flatten_bs):
#             print(f"batches1[i]: {batches1[i].shape}, batch2: {batch2.shape}")
#             batch.append(batches1[i] @ batch2)
    
#     out_shape = list(batch1_shape[:-2])
#     out_shape += [d_model, hiden_dim]
#     b_mm = stack(batch, axis=0).reshape(out_shape)
#     return b_mm
    
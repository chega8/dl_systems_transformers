import sys
sys.path.append('./python')

import numpy as np
import pytest
from needle import backend_ndarray as nd
import needle as ndl


def test_softmax():
    a = np.random.random((4, 3))
    
    np_logits = np.array(a)
    
    np_logits = np.exp(np_logits - np_logits.max(axis=-1, keepdims=True))
    np_softmax = np_logits / np_logits.sum(axis=-1, keepdims=True)
    
    
    ndl_logits = ndl.Tensor(a)
    ndl_softmax = ndl.functional.softmax(ndl_logits)
    
    err = np.linalg.norm(np_softmax - ndl_softmax.numpy())
    assert err < 1e-3
    
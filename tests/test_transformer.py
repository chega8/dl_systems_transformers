import sys
sys.path.append('./python')

import numpy as np
import torch
import pytest

from needle import backend_ndarray as nd
import needle as ndl


def test_batch_mm_pass():
    b1_shape = (10, 15, 4, 5)
    b2_shape = (10, 15, 5, 6)

    a = np.random.random(b1_shape).astype(np.float32)
    b = np.random.random(b2_shape).astype(np.float32)

    ndl_a = ndl.Tensor(a, device=ndl.cpu(), requires_grad=True)
    ndl_b = ndl.Tensor(b, device=ndl.cpu(), requires_grad=True)

    torch_a = torch.tensor(a, requires_grad=True, dtype=torch.float32)
    torch_b = torch.tensor(b, requires_grad=True, dtype=torch.float32)

    ndl_bmm = ndl_a @ ndl_b
    loss_ndl = ndl_bmm.sum()
    loss_ndl.backward()

    torch_bmm = torch_a @ torch_b
    loss = torch_bmm.sum()
    loss.backward()

    err = np.linalg.norm(torch_bmm.detach().numpy() - ndl_bmm.detach().numpy())
    assert err < 1e-3
    
    # check grads
    err = np.linalg.norm(torch_a.grad.numpy() - ndl_a.grad.numpy())
    assert err < 1e-3
    err = np.linalg.norm(torch_b.grad.numpy() - ndl_b.grad.numpy())
    assert err < 1e-3
    
    b1_shape = (10, 32, 16)
    b2_shape = (16, 10)

    a = np.random.random(b1_shape).astype(np.float32)
    b = np.random.random(b2_shape).astype(np.float32)

    ndl_a = ndl.Tensor(a, device=ndl.cpu(), requires_grad=True)
    ndl_b = ndl.Tensor(b, device=ndl.cpu(), requires_grad=True)

    torch_a = torch.tensor(a, requires_grad=True, dtype=torch.float32)
    torch_b = torch.tensor(b, requires_grad=True, dtype=torch.float32)

    ndl_bmm = ndl_a @ ndl_b
    loss_ndl = ndl_bmm.sum()
    loss_ndl.backward()

    torch_bmm = torch_a @ torch_b
    loss = torch_bmm.sum()
    loss.backward()

    err = np.linalg.norm(torch_bmm.detach().numpy() - ndl_bmm.detach().numpy())
    assert err < 1e-3
    
    # check grads
    err = np.linalg.norm(torch_a.grad.numpy() - ndl_a.grad.numpy())
    assert err < 1e-3
    err = np.linalg.norm(torch_b.grad.numpy() - ndl_b.grad.numpy())
    assert err < 1e-3
    
    
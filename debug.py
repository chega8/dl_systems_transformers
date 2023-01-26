import sys
sys.path.append('./python')

from needle import backend_ndarray as nd
import numpy as np

import needle as ndl
import needle.ops as ops
import numpy as np
import torch

b1_shape = (2, 3, 5)
b2_shape = (5, 4)

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

np.linalg.norm(torch_bmm.detach().numpy() - ndl_bmm.detach().numpy())
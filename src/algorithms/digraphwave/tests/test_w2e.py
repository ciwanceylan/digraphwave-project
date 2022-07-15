import pytest
import numpy as np
import torch
import torch_sparse as tsp

import digraphwave.wavelets2embeddings as w2e


class TestTransform:

    def test_transform_log_arctan(self):
        a_tensor = torch.tensor([0, 1, 10, torch.inf])
        a_trans = w2e.transform_dense(a_tensor, maxval=None, minval=0, log_arctan=True)
        assert a_trans[0] == 0
        assert a_trans[1] == pytest.approx(np.arctan(np.log(1)) + 0.5 * np.pi)
        assert a_trans[2] == pytest.approx(np.arctan(np.log(10)) + 0.5 * np.pi)
        assert a_trans[3] == pytest.approx(np.pi)

    def test_transform_log_arctan_with_maxval(self):
        a_tensor = torch.tensor([0, 1, 10])
        a_trans = w2e.transform(a_tensor, maxval=10, minval=0, log_arctan=True)
        assert a_trans[0] == 0
        assert a_trans[1] == pytest.approx(np.arctan(np.log(np.tan(np.pi / 20))) + 0.5 * np.pi)
        assert a_trans[2] == pytest.approx(np.arctan(np.log(np.tan(np.pi / 2))) + 0.5 * np.pi)

    def test_transform(self):
        a_tensor = torch.tensor([0, 1, 10])
        a_trans = w2e.transform(a_tensor, maxval=10, minval=0, log_arctan=False)
        assert a_trans[0] == 0
        assert a_trans[1] == pytest.approx(np.pi / 10)
        assert a_trans[2] == pytest.approx(np.pi)

    def test_transform_sparse_log_arctan_inplace(self):
        a_tensor = torch.tensor([[0, 0, 0, 1, 10, torch.inf]])
        a_tensor = tsp.SparseTensor.from_dense(a_tensor)
        a_trans = w2e.transform(a_tensor, maxval=None, minval=0, log_arctan=True)
        assert a_tensor == a_trans
        a_trans = a_trans.to_dense()
        assert a_trans[0, 0] == 0
        assert a_trans[0, 1] == 0
        assert a_trans[0, 2] == 0
        assert a_trans[0, 3] == pytest.approx(np.arctan(np.log(1)) + 0.5 * np.pi)
        assert a_trans[0, 4] == pytest.approx(np.arctan(np.log(10)) + 0.5 * np.pi)
        assert a_trans[0, 5] == pytest.approx(np.pi)

    def test_transform_sparse_log_arctan(self):
        a_tensor = torch.tensor([[0, 0, 0, 1, 10, torch.inf]])
        a_tensor = tsp.SparseTensor.from_dense(a_tensor)
        a_trans = w2e.transform_sparse(a_tensor, maxval=None, minval=0, log_arctan=True, copy=True)
        assert not torch.all(a_tensor.storage.value() == a_trans.storage.value())
        a_trans = a_trans.to_dense()
        assert a_trans[0, 0] == 0
        assert a_trans[0, 1] == 0
        assert a_trans[0, 2] == 0
        assert a_trans[0, 3] == pytest.approx(np.arctan(np.log(1)) + 0.5 * np.pi)
        assert a_trans[0, 4] == pytest.approx(np.arctan(np.log(10)) + 0.5 * np.pi)
        assert a_trans[0, 5] == pytest.approx(np.pi)


def test_thresholding():
    a_tensor = torch.tensor(
        [[0, 1, 2],
         [1, 2, 0],
         [2, 0, 1]]
    )
    theta = torch.tensor([1, 2, 0])
    result = w2e.threshold_batch_inplace(a_tensor, theta, as_sparse=False)
    assert torch.equal(result, a_tensor)
    print(result)
    assert torch.all(
        result == torch.tensor(
            [[0, 0, 2],
             [1, 2, 0],
             [2, 0, 1]]
        )
    )


def test_wavelets2embeddings():
    theta = torch.arange(0, 0.1, 10, dtype=torch.float64)
    a_tensor = w2e.threshold_batch_inplace(torch.randn(100, 10, dtype=torch.float64), theta, as_sparse=False)
    a_tensor_sp = tsp.SparseTensor.from_dense(a_tensor)
    time_points = torch.arange(0.5, 10, 0.1)
    res1 = w2e.wavelets2embeddings(a_tensor, time_points)
    res2 = w2e.wavelets2embeddings(a_tensor_sp, time_points)
    assert torch.max(res1 - res2) < 1e-14

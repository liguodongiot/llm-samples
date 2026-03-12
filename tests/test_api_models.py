import pytest
from matmul_ops.api.models import MatrixInput, MatrixOutput


def test_matrix_input_valid():
    data = {"A": [[1.0, 2.0], [3.0, 4.0]], "B": [[5.0, 6.0], [7.0, 8.0]]}
    model = MatrixInput(**data)
    assert model.A == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_output_valid():
    data = {"result": [[19.0, 22.0], [43.0, 50.0]]}
    model = MatrixOutput(**data)
    assert model.result == [[19.0, 22.0], [43.0, 50.0]]

from typing import Tuple, Union

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

import numpy as np


def refresh(size: int, qreg_name: str = 'qr', creg_name: str = 'cr', circ_name: str = 'qc') \
        -> Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]:
    qr = QuantumRegister(size, qreg_name)
    cr = ClassicalRegister(size, creg_name)
    qc = QuantumCircuit(qr, cr, name=circ_name)

    return qc, qr, cr


def real_matrix(mx: Union[list, np.array]) -> np.array:
    return np.array([[x.real for x in row] for row in mx])


def norm_matrix(mx: Union[list, np.array], norm: float) -> np.array:
    return np.array([[(x / norm) for x in row] for row in mx])


def cleanup_matrix(mx: Union[list, np.array], norm: float) -> np.array:
    cleaned = norm_matrix(real_matrix(mx), norm)
    cleaned = [[int(x) for x in row] for row in cleaned]
    cleaned = str(cleaned).replace('\n', '').replace('   ', '  ').replace(']', ']\n')
    return np.array(cleaned)


def trim_matrix(mx: Union[list, np.array], dim: int) -> np.array:
    """ Takes an nxn matrix and returns a reduced matrix of dimension dim x dim
    corresponding to the upper-left square of the original matrix"""
    trimmed = [[x for idx, x in enumerate(row) if idx < dim] for index, row in enumerate(mx) if index < dim]
    return np.array(trimmed)

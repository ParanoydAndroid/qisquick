import logging

from typing import Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def refresh(size: int, qreg_name: str = 'qr', creg_name: str = 'cr', circ_name: str = 'qc') \
        -> Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]:
    qr = QuantumRegister(size, qreg_name)
    cr = ClassicalRegister(size, creg_name)
    qc = QuantumCircuit(qr, cr, name=circ_name)

    return qc, qr, cr

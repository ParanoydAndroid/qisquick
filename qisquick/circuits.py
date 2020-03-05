from __future__ import annotations

import math
import random
import time

import numpy as np

import qisquick.dbconfig as dbc

from datetime import datetime
from typing import List, Union

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, transpile, execute, Aer
from qiskit.providers import basebackend
from qiskit.providers.jobstatus import JobStatus
from qiskit.transpiler import PassManager

from qisquick.run_experiment import PREFERRED_BACKEND
from qisquick.statblock import Statblock
from qisquick.qis_logger import get_module_logger

logger = get_module_logger(__name__)
preferred_backend: str = PREFERRED_BACKEND


class Premades(QuantumCircuit):
    def __init__(self, size: int, truth_value: int, measure: bool = True, seed: int = None):
        """ Creates a Premades object that wraps QuantumCircuits to carry additional information.  Most important is
            that the PreMades object stores the uniform interface parameters for generating new circuits.

        Args:
            size (int): Width of the desired circuit.  i.e. the register size of the quantum register defining it.
            truth_value (int): An inteeger to encode in any oracles that the circuit uses.  Usually used to define
                the "right" value for the circuit to return.  E.g. the correct value for a grover's search to find.
            measure (bool): Optional. If True, adds measurement operators to the end of the circuit.
            seed (int): Optional.  If not None, the provided seed is used to set random state for reproducibility.
        """
        if truth_value == 0:
            logger.warning('Truth values that evaluate to basis 00 ... 00 may cause misleading accuracy measurements')

        qr = QuantumRegister(size, 'qr')
        cr = ClassicalRegister(size, 'cr')
        super().__init__(qr, cr, name='qc')

        self.circ_size = size
        self.truth_value = truth_value
        self.meas = measure
        self.seed = seed

    @property
    def qr(self):
        return self.qregs[0]

    @property
    def cr(self):
        return self.cregs[0]

    def two_bell(self) -> None:
        size = self.circ_size
        random.seed(self.seed)

        if size % 2 != 0:
            # TODO just change this to allow odd qubits
            raise ValueError(f'Circuit must be constructed with an even number of qubits; was given {size}')

        # Make the program square by using the same number of layers as there are pairs.
        for i in range(size // 2):
            pairings = np.random.permutation(range(size))
            for j in range(size // 2):
                self.h(pairings[2 * j])
                self.cx(pairings[2 * j], pairings[2 * j + 1])

        if self.meas: self.measure(self.qr, self.cr)

    def m_to_m(self) -> None:
        size = self.circ_size
        random.seed(self.seed)
        candidates = tuple(range(size))

        self._get_random_input_state(self.seed)

        for i in range(self.circ_size):  # TODO assess whether the condition should be static or 'size'
            for candidate in candidates:
                num_connections = random.randint(0, 5)
                connections = set()
                for j in range(num_connections):
                    partner = random.choice(candidates)
                    while partner == candidate:
                        partner = random.choice(candidates)

                    connections.add(partner)

                for connection in connections:
                    self.cx(candidate, connection)

        if self.meas: self.measure(self.qr, self.cr)

    def grover(self) -> None:
        """ Create and return a quantum circuit implementation of Grover's search algorithm. """

        size = self.circ_size
        random.seed(self.seed)

        # Split the single qreg into the computational register and a "work" register to store the oracle results
        qr = self.qr[:-1]
        wr = self.qr[-1:]
        tv = self.truth_value

        self.h(qr)

        loop = math.floor(math.sqrt(2 ** size))
        logger.debug(f'loop: {loop}')
        for i in range(loop):
            # NOT the wires where the magic state bit should be 0
            for j in range(len(qr)):
                if not tv & (1 << j):  # i.e. tharr be zeroes in this here bitstring position
                    self.x(qr[j])

            self.x(wr[0])
            self.mcrz(math.pi, qr, wr[0])
            self.x(wr[0])

            for j in range(len(qr)):
                if not tv & (1 << j):  # i.e. tharr be zeroes in this here bitstring position
                    self.x(qr[j])

            # Inversion about the average
            self.h(qr)
            self.x(qr)

            self.mcrz(math.pi, qr[:-1], qr[-1])

            self.x(qr)
            self.h(qr)

        if self.meas: self.measure(self.qr, self.cr)

    def islands(self) -> None:
        size = self.circ_size

        self._get_random_input_state(self.seed)

        # We use the islands set to ensure different qubits are picked as the 'hubs' across layers.
        # This is distinct from the 'used' set which just ensures we don't pick the same qubit twice in the same layer.
        islands = set()

        for j in range(size):
            candidates = set(range(size))
            used = set()
            qubits_used = 0
            next_layer = False  # Flag for when all possible partners have been consumed for this layer
            for candidate in candidates:
                if candidate in used:
                    continue
                if candidate in islands:
                    islands.remove(candidate)
                    continue
                else:
                    used.add(candidate)
                    islands.add(candidate)
                    qubits_used += 1

                num_connections = random.randint(0, size - 1)

                # This procedure might leave one, isolated qubit at the end of the loop.
                if qubits_used + num_connections > size:
                    num_connections = size - qubits_used
                    next_layer = True
                else:
                    qubits_used += num_connections

                connections = set()
                for i in range(num_connections):
                    partner = random.choice(tuple(candidates.difference(used)))
                    connections.add(partner)
                    used.add(partner)

                for connection in connections:
                    self.cx(candidate, connection)

                if next_layer: break

        if self.meas: self.measure(self.qr, self.cr)

    def uniform_random(self) -> None:
        """ Creates a circuit whose gates are uniformly chosen from {H, X, Y, Z, S, T, CX}
        and whose CX endpoints are chosen uniformly from available qubits"""

        size = self.circ_size
        random.seed(self.seed)

        gates = [self.h, self.x, self.y, self.z, self.s, self.t, self.cx]
        candidates = set(range(size))

        for i in range(size):
            for j in range(size):
                to_apply = random.choice(gates)

                num_qubits = 2 if to_apply == self.cx else 1
                targets = random.sample(candidates, num_qubits)
                to_apply(*targets)

        if self.meas: self.measure(self.qr, self.cr)

    def bv(self) -> None:
        """ Implements a Bernstein-Vazirani algorithm circuit, where the oracle encodes self.truth_value.

        Returns:
            None:
        """
        size = self.circ_size
        random.seed(self.seed)
        meas = self.meas
        tv = self.truth_value

        if math.ceil(math.log2(tv)) + 1 > size:
            raise ValueError(f'Truth value too large to be represented in {size} qubits.  Circuit must have at least\
                             size {math.ceil(math.log2(tv)) + 1}')

        # Setup initial superposition to feed and the phase kickback:
        self.x(self.qr[-1])
        self.h(self.qr)

        # Encode tv oracle
        for i in range(len(self.qr[:-1])):
            if tv & (1 << i):
                self.cx(self.qr[i], self.qr[-1])

        self.h(self.qr)

        if meas: self.measure(self.qr, self.cr)

    def toff(self) -> None:
        """ Create a series of toffoli gates across the size of the circuit.
        If not 3 | size, then some qubits will not be used in each layer"""

        size = self.circ_size
        self._get_random_input_state(self.seed)

        remainder = size % 3
        for layer in range(size):
            leftovers = tuple(random.sample(range(size), remainder))
            candidates = [x for x in range(size) if x not in leftovers]

            while candidates:
                current_toff = tuple(random.sample(candidates, 3))
                self.ccx(current_toff[0], current_toff[1], current_toff[2])
                candidates = [x for x in candidates if x not in current_toff]

        self.decompose().decompose()
        if self.meas: self.measure(self.qr, self.cr)

    def qft(self) -> None:
        """ implementation stolen shamelessly from github.com/Qiskit/qiskit-terra/blob/master/examples/python/qft.py"""
        random.seed(self.seed)
        self.__qft_input_state(self.circ_size)

        for i in range(self.circ_size):
            for j in range(i):
                self.cu1(math.pi / float(2 ** (i - j)), i, j)
                self.h(i)

        if self.meas: self.measure(self.qr, self.cr)

    def __qft_input_state(self, size) -> None:
        """Encodes the truth_value into the circuit for later recovery by the QFT algo"""

        for i in range(size):
            self.h(i)
            self.u1(-math.pi / float(2 ** i), i)

    def _get_random_input_state(self, seed: int = None) -> None:
        """ Randomizde the input state of a circuit"""

        random.seed(seed)
        size = self.circ_size
        # Determine how many qubits to set to the 1 state
        num_winners = random.randint(1, size)

        # Now that we know how many, which ones?
        winners = random.sample(range(size), num_winners)

        for winner in winners:
            self.x(winner)

    circ_lib = {
        'two_bell': two_bell,
        'm_to_m': m_to_m,
        'grover': grover,
        'moving_island': islands,
        'uniform_random': uniform_random,
        'bv': bv,
        'toffoli': toff,
        'qft': qft
    }


class TestCircuit:
    def __init__(self):
        self.stats = Statblock(parent=self)
        self.compiled_circ = None
        self.backend = None
        self.job_id = None
        self.transpiler_config = None
        self.circuit = None

        # if isinstance(circuit, QuantumCircuit):
        #     self.circuit = circuit
        # elif circuit is not None:
        #     raise TypeError(f'Circuit must be a QuantumCircuit, or Premade.  Was given type: {type(circuit)}')

    @property
    def id(self):
        return self.stats.id

    @id.setter
    def id(self, new_id):
        self.stats.id = new_id

    @property
    def job(self):
        if self.job_id is None or self.backend is None:
            bad_thing = 'job_ID' if self.job_id is None else 'backend'
            logger.warning(f'Requested job object on TestCircuit that has no job. {bad_thing} is None')
            return None
        else:
            backend = self.get_circ_backend()
            return backend.retrieve_job(self.stats.job_id)

    @staticmethod
    def generate(case: str, size: int, measure: bool = True, seed: int = None) -> TestCircuit:
        tc = TestCircuit()
        tc.add_circ(case, size, measure=measure, seed=seed)

        return tc

    @staticmethod
    def run_all_tests(tests: Union[List[TestCircuit], List[QuantumCircuit], TestCircuit, QuantumCircuit],
                      pass_manager: Union[PassManager, List[PassManager]] = None, generate_compiled: bool = True,
                      be: str = preferred_backend, attempts: int = 1) -> None:
        """ Given a circuit or list of circuits to execute, it executes all of them and writes all results to the
            appropriate db.  Depending on parameters, a custom PassManager can be used, and the circuits will also be
            compiled before execution.

        Args:
            tests (List[TestCircuit]): Circuits to be tested
            pass_manager (PassManager): Custom PassManager to use for transpilation, if desired.  Default: IBM default
            generate_compiled (bool): If True, will transpile circuits prior to execution
            be (Backend): IBM backend to use for transpilation and execution.  Default: preferred_backend
            attempts: Number of times to transpile the circuits to generate average compile time

        Returns: None (but writes results to statistics database as a side effect)

        """

        if not isinstance(tests, List): tests = [tests]
        if len(tests) > 25:
            logger.warning(f'Batch size might exceed maximum.  Currently {len(tests)}')

        # If the circuits have been separately transpiled, we need to ensure they were done so uniformly
        compiled_circs = []
        if not generate_compiled:
            if len({tc.backend for tc in tests}) != 1:
                raise ValueError(f'All circuits in the same batch must use the same backend.')

            compiled_circs = [tc.compiled_circ for tc in tests]
            if None in compiled_circs:
                raise ValueError(f'Test Run failed on batch (first id: {tests[0].id}).  '
                                 f'No transpiled circuits available.  '
                                 f'Set generate_compiled=True to have this done automatically')
        else:
            # If a a list of PassManagers of the same len() as tests was provided, we're good.  Otherwise listify.
            if not isinstance(pass_manager, List):
                pass_manager = [pass_manager for t in tests]

            elif len(pass_manager) != len(tests):
                raise IndexError(f'Error in function run_all_tests: Mismatch in len(tests) && len(pass_manager)')

            for idx, tc in enumerate(tests):
                tc.backend = be
                tc.transpile_test(pass_manager=pass_manager[idx], default_be=be, ATTEMPTS=attempts)
                compiled_circs.append(tc.compiled_circ)
                tc.stats.iteration = idx

        dbc.write_objects(dbc.db_location, tests)
        job = execute(compiled_circs, backend=tests[0].get_circ_backend())
        for tc in tests:
            tc.get_ideal_result()
            tc.job_id = job.job_id()

        dbc.insert_in_progress(dbc.db_location, tests)
        dbc.write_objects(dbc.db_location, tests)

    def get_ideal_result(self):
        sim = Aer.get_backend('qasm_simulator')
        self.stats.ideal_distribution = execute(self.compiled_circ, sim).result().get_counts()

    def add_circ(self, case: Union[str, Premades], size: int, truth_value: int = None, measure=True, seed: int = None) \
            -> None:
        """ Given an empty TestCircuit, add a circuit to it.

        Args:
            case (str): Dictionary key corresponding to circuit-generating function OR an existing circuit of type
                Premades
            size (int): width of circuit, in qubits.  If case is a str, a circuit of size will be created.  If case
                is a PreMades, then size should match the existing Premades circ_size attribute.
            truth_value (Union[int, None]): Integer corresponding to basis vector that should be
                returned by the circuit when executed on an ideal simulator.  If case is a str, then truth_value will
                  be encoded in any oracles created by Premades circuit functions. If case is a PreMades, then
                  truth_value should match the existing Premades truth_value attribute.
            measure (bool): Optional. Add measurement operators to created circuit.  Unused if case if of type
                Premades
            seed (int): Optional. Sets Random state for reproducibility.  Unused if case is of type Premades.

        Returns:
            None:
        """

        # TODO Refactor this to more elegantly accept either type for case.  e.g. by splitting into two methods
        #  (don't like that option), or maybe by changing to a **kwargs format, though that will obscure inner workings.

        if isinstance(case, str):
            self.circuit = Premades(size, truth_value, measure=measure, seed=seed)
            self.circuit.circ_lib[case](self.circuit)

            self.stats.name = case
            self.stats.truth_value = truth_value
            self.stats.circ_width = size
            self.stats.seed = seed

        elif isinstance(case, Premades):
            # If we're being given an existing circ, we use it for our single source of truth and just ignore
            # The passed parameters
            self.circuit = case
            self.stats.name = self.circuit.name
            self.stats.truth_value = self.circuit.truth_value
            self.stats.circ_width = self.circuit.circ_size
            self.stats.seed = self.circuit.seed
        else:
            raise ValueError(f'Case must be either a key reference to existing circuit generators or a Premades')

        self.stats.pre_depth = self.circuit.depth()

    def transpile_test(self, pass_manager=None, default_be=preferred_backend, ATTEMPTS: int = 1) -> QuantumCircuit:
        """ Transpile TestCircuit with provided pass_manager and register statistics, but do not execute.

        Args:
            pass_manager (PassManager): Custom PassManager to use to transpile this circuit.
            default_be (str): Optional. Default backend to use for transpilation; defaults to preferred_backend defined
                in run_experiment.py
            ATTEMPTS (int): Optional. Number of transpile tests to be run to generate averages.

        Returns:
            qiskit.circuit.quantumcircuit.QuantumCircuit: Returns the compiled circuit for chaining; also saves it to
                self.compiled_circ as a side-effect.
        """

        if self.backend is None:
            logger.warning(f'Transpiler: Circuit ({self.id}) had no backend.  Resorted to default: {preferred_backend}')
            self.backend = default_be

        transpile_times = []

        # Get the average transpile time over ATTEMPTS number of trials
        for i in range(ATTEMPTS):
            start_time = time.process_time()
            self.compiled_circ = transpile(self.circuit,
                                           backend=self.get_circ_backend(),
                                           optimization_level=0,
                                           pass_manager=pass_manager)
            transpile_times.append(time.process_time() - start_time)

        tc: QuantumCircuit = self.compiled_circ
        stats = self.stats

        logger.info(f'Transpiled and registered {self.stats.name}: {self.id}')

        # Returns average in ms
        stats.compile_time = (sum(transpile_times) * (10 ** 3)) / len(transpile_times)
        stats.post_depth = tc.depth()

        logger.info(f'Transpiled circ of depth {stats.post_depth} in {stats.compile_time}ms.')

        pre_cx = 0
        post_cx = 0
        if 'cx' in self.circuit.count_ops().keys():
            pre_cx = self.circuit.count_ops()['cx']
        if 'cx' in tc.count_ops().keys():
            post_cx = tc.count_ops()['cx']

        stats.swap_count = (post_cx - pre_cx) / 3

        dbc.write_objects(dbc.db_location, [self])

        return tc

    def run_job(self):
        """ Executes self.compiled_circ on self.backend and register it as a running job in the Running table

        Returns:
            None:
        """
        be = self.get_circ_backend()

        if self.compiled_circ is None:
            raise ValueError(f'Cannot run circuit {self.stats.name}: {self.id}.  No compiled version available')

        self.get_ideal_result()
        job = execute(self.compiled_circ, be)
        self.job_id = job.job_id()
        dbc.write_objects(dbc.db_location, [self])
        dbc.insert_in_progress(dbc.db_location, [self])

    def get_circ_backend(self, hub: str = 'ibm-q-afrl', default_backend=preferred_backend) -> basebackend:
        """ Helper function to map a backend's string ID to its object.

        Args:
            hub (str): Provider owning the backend
            default_backend (str): String identifier of backend

        Returns:
            qiskit.providers.ibmq.ibmqbackend.IBMQBackend:
        """
        default_backend = self.backend if self.backend is not None else default_backend
        return IBMQ.get_provider(hub=hub).get_backend(default_backend)

    def get_post_stats(self):
        """ Retrieves, but does not store, results from the execution of this TestCircuit.
        """
        stats = self.stats
        stats.results = self.job.result().get_counts(stats.iteration)
        stats.datetime = str(datetime.now())

    def get_status_done(self) -> bool:
        return self.job.status() is JobStatus.DONE

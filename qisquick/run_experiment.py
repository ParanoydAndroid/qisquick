import argparse
import logging
import sqlite3
import sys
import traceback

from time import sleep
from typing import List, Union, Any, Callable

from matplotlib.pyplot import close
from qiskit import *
from qiskit.transpiler import CouplingMap
from qiskit.visualization import circuit_drawer


from qls import dbconfig as dbc
from qls import Analysis, transpilertools
from qls.circuits import TestCircuit, Premades
from qls.qis_logger import config_logger, get_module_logger

PREFERRED_PROVIDER = None
PREFERRED_BACKEND = None
logger = None
backend = None
args = None


def main(argv=None, experiment: Callable = None):
    """ Main function.  Called automatically on execution, or can be called indirectly via the run_experiment function
        if this module is imported

    Args:
        argv (argparse.Namespace): Optional. CLI arguments.  Only used if being run as actual main
        experiment (Callable): Optional. Experiment to execute.  Only used if main is called from run_experiment().
            The provided experiment should accept no arguments and should return a list of database IDs to check
            for circuit execution completion.

    Returns:
        None:
    """
    run_once = True
    ids = []

    if argv is not None:
        check_only = argv.check_only
    else:
        check_only = False

    experiment_to_run = experiment if experiment is not None else run_local_experiment
    while True:
        if run_once and not check_only:
            ids = experiment_to_run()
            run_once = False

        logger.info(f'\t\tCHECKING FOR COMPLETED CIRCUITS...')
        check_running()
        if dbc.is_empty(dbc.db_location, 'Running'):
            dbc.write_stats(dbc.db_location, ids)
            logger.info(f'All Running jobs have completed.  Shutting down ...')
            break

        # seconds
        sleep(60)

    sys.exit(0)


def run_experiment(experiment: Callable,  db_location: str = None, provider: str = None, backend: str = None) -> None:
    """ Function provided for run_experiment.py imports into other systems.  This can be called with a user
        defined experiment script and will perform the same tasks as if main() was called on that function

    Args:
        experiment (Callable): The function defined by the user to run their experiment
        db_location (str): If this function is being called from an import, we assume the db location might be
            different, so let's ensure it's provided.  Should be of the form 'relative/path/dbName.sqlite'
        provider (str): Optional. String reference used to retrieve provider object.  Defaults to AFRL Hub (private)
        backend (str): Optional. String reference used to retrieve backend. Defaults to IBMQ_poughkeepsie (private)


    Returns: None.  But as a side-effect will write to the Circs and Stats tables at db_location.
    """

    pre_process(provider, backend)
    dbc.set_db_location(db_location)
    dbc.create_all_tables(dbc.db_location)
    main(None, experiment)


def run_local_experiment() -> List[str]:
    """By Brandon Kamaka, 30 Jan 2020.  Reproducibility experiment to validate test bed
        Create a series of test circuits, and transpile each series with distinct options from various layout and SWAP
        optimizing papers.  Compare success, SWAP efficiency, and time efficiency metrics
    """

    dbc.set_db_location('data/circuit_data.sqlite')
    pass_configurations = {
        0: 'IBM Baseline',
        1: 'Lookahead SWAP',
        2: 'Noise-Adaptive (GreedyE)',
        3: 'IBM Optimized'
    }

    circuits_to_test = {
        0: 'two_bell',
        1: 'uniform_random',
        2: 'bv',
        3: 'qft',
        4: 'grover'
    }

    num_trials = 50
    test_config = (3, 4)  # Experiment details, circuit

    logger.info(f'+++++++++++++++++TEST CONFIG: {test_config}++++++++++++++++++++++++++++++++++')

    # ************************** Phase 1: Make initial circuits and prep
    case = circuits_to_test[test_config[1]]
    filename = f'{pass_configurations[test_config[0]]} - {case}'

    # Make the Premades object.  It does not contain a circuit but stores the uniform information for circuit creation.
    circ = Premades(size=3, truth_value=2, measure=True)

    # Actually add a specific QuantumCircuit instance based on the stored parameters
    Premades.circ_lib[case](circ)
    circ.draw(output='mpl',
              filename=filename)
    tests = []
    for i in range(num_trials):
        # Create distinct TestCircuit objects (so that each gets its own unique ID), but each TC gets the same PreMades
        tc = TestCircuit()
        tc.add_circ(circ, size=3, truth_value=2, measure=True)
        tc.stats.name = case
        tc.stats.notes = filename + f' - {i}'
        tests.append(tc)

    # Register initial statistics
    dbc.write_objects(dbc.db_location, tests)

    # ************************** Phase 2, make the distinct passmanagers for each test config and circuit

    # Start by getting a transpiler config from the circuits and backend
    level = 1 if pass_configurations[test_config[0]] != 'IBM Optimized' else 3
    configs = transpilertools.get_transpiler_config(circs=tests, be=backend, optimization_level=level)

    # Then we use the configs to get the appropriate PassManager for each configuration
    pms = []
    for idx, config in enumerate(configs):
        pm = transpilertools.get_basic_pm(config, level=level)
        modified_pm = transpilertools.get_modified_pm(pass_manager=pm, backend=backend, version=test_config[0])
        pms.append(modified_pm)

    logger.info(f'++++++++++++++++++++++++PM BEING USED++++++++++++++++++++++++++++++++++')
    logger.info(transpilertools.get_passes_str(pms[0]))
    logger.info(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # ************************** Phase 3: Run tests on circuits with custom PassManagers

    # Just in case our number of test_cases exceeds a reasonable size (25)
    test_batches = get_batches(tests)
    pms_batches = get_batches(pms)
    assert len(test_batches) == len(pms_batches)
    for test_batch, pms_batch in zip(test_batches, pms_batches):
        TestCircuit.run_all_tests(test_batch, pass_manager=pms_batch, be=PREFERRED_BACKEND, attempts=1)

    # ************************** Phase 4: Return final circ ids so the normal routine can save them to Stats
    return [test.id for test in tests]


def do_once():
    """Functions and code that should not be endlessly looped should be put here."""
    # run_local_experiment()
    # circs = [TestCircuit.generate('two_bell', 4) for i in range(4)]
    #
    # pms = []
    # for i in range(4):
    #     level = 1 if i < 3 else 3
    #     tcs = TranspilerTools.get_transpiler_config(circs[i], be=backend, optimization_level=level)
    #
    #     pm = TranspilerTools.get_basic_pm(tcs[0], level)
    #     pm = TranspilerTools.get_modified_pm(pm, backend, i)
    #     pms.append(pm)
    #
    # for pm in pms:
    #     print('\nNEW PM')
    #     TranspilerTools.get_passes_str(pm)


def get_batches(tcs: List[Any], batch_size: int = 25) -> List[List[Any]]:
    """ IBM QX devices expect relatively small lists of circuits to run.  This function takes a list of circuits
        of arbitrary size and returns a list of list of circuits, where each inner list contains batch_size circuits.
    Args:
        tcs (List): List to be batched
        batch_size (int):

    Returns:
        List[List[Any]]: Original list of circuits, split into batch_size
    """
    is_testcircuit = False
    if isinstance(tcs[0], TestCircuit): is_testcircuit = True

    batches = []
    while len(tcs) > batch_size:
        batch = tcs[:batch_size]

        # Add iteration variable so we know which count corresponds with which circuit
        if is_testcircuit:
            for idx, tc in enumerate(batch):
                tc.stats.iteration = idx

        batches.append(batch)
        tcs = tcs[batch_size:]

        # We enter this loop only when we're at the end of the list and have < batch_size circuits remaining
    if is_testcircuit:
        for idx, tc in enumerate(tcs):
            tc.stats.iteration = idx

    batches.append(tcs)
    return batches


def check_running():
    """ Checks the "Running" table of the linked db to see if any batches in progress have been executed by the IBM
        backend.  If so, it retrieves the job details, saves them to the main stats db and deletes the record from
        Running.
    """
    completed = {}
    try:
        running = dbc.load_in_progress(dbc.db_location)
        checked = {}
        for k, v in running.items():
            # Avoid re-polling already checked jobs, since we can have multiple circuits under the same job_id.
            if v.job_id in checked.keys():
                done = checked[v.job_id]
            else:
                done = checked[v.job_id] = v.get_status_done()

            if done:
                v.get_post_stats()
                completed[k] = v

        if len(completed):
            print(f'Completed {len(completed)} jobs.')
            dbc.write_objects(dbc.db_location, list(completed.values()))
            dbc.drop_in_progress(dbc.db_location, list(completed.keys()))
    except Exception:
        msg = 'Error retrieving jobs in progress from last session'
        logger.error(f'{msg}: {traceback.format_exc()}')
    else:
        if len(completed):
            msg = f'The following jobs completed and were properly registered:\n'
            for _id in completed:
                msg += f'\tjob: {completed[_id].job_id} - ID: {_id}\n'

            logger.info(f'{msg}')


def create_all(size: int = 4, truth_value: int = 5, filename: str = None) -> List[TestCircuit]:
    """ Quick functionality test.  Creates and returns one copy of each test set object, and generates diagrams of them.

    Args:
        size (int): Width of circuit to generate.  Sizes > ~14 qubits are likely to take a long time.
        truth_value (int): Desired base state for the circuit to return under measurement, if executed on an ideal sim.
            This is not applicable for all test circuits (e.g. QFT)
        filename (str): If given, will cause each generated circuit to create a .png of its composer format.

    Returns:
        List[qls.circuits.TestCircuit]:
    """
    circs = []
    for key in Premades.circ_lib.keys():
        new_test = TestCircuit()
        logger.info(f'Creating test circuit for {key} function')
        new_test.add_circ(key, size=size, truth_value=truth_value, measure=True)
        circs.append(new_test)
        print(f'Test circuit ({key}) stats (pre-transpile):\n{str(new_test.stats)}')
        if filename:
            fig = new_test.circuit.draw(output='mpl', filename=f'{filename + "-" + key}.png')
            fig.suptitle(f'{key}', fontsize=16)
            close(fig)

    return circs


def get_cli_args():
    parser = argparse.ArgumentParser(description="driver for testing layout scheme")
    parser.add_argument('--verbose', '-v', action='count', default=None, help='Increase logging level, up to -vvv')
    parser.add_argument('--check-only', '-C', action='store_true',
                        help='If -C, then the program will not generate circuits, '
                             'and will only check for job updates for previously submitted jobs.')
    return parser.parse_args()


def pre_process(default_provider: str = None, default_backend: str = None) -> None:
    global PREFERRED_PROVIDER, PREFERRED_BACKEND, logger, backend, args

    PREFERRED_BACKEND = default_backend if default_backend is not None else 'ibmq_poughkeepsie'
    PREFERRED_PROVIDER = default_provider if default_provider is not None else 'ibm-q-afrl'
    args = get_cli_args()
    config_logger(args.verbose)
    logger = get_module_logger(__name__)
    logger.critical('Showing critical messages')
    logger.error('Showing error messages')
    logger.warning('Showing warning messages')
    logger.info('Showing informational messages')
    logger.debug('Showing debug messages')
    IBMQ.load_account()
    provider = IBMQ.get_provider(PREFERRED_PROVIDER)
    backend = provider.get_backend(PREFERRED_BACKEND)


if __name__ == '__main__':
    pre_process()
    main(args)

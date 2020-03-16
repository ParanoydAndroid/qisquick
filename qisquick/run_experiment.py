import argparse
import sys
import traceback

import qisquick.circuits as qqc

from time import sleep
from typing import List, Any, Callable

from qiskit import IBMQ

from qisquick import dbconfig as dbc
from qisquick.circuits import Premades, TestCircuit
from qisquick.qis_logger import config_logger, get_module_logger

PREFERRED_PROVIDER_NAME = 'ibm-q'
PREFERRED_BACKEND_NAME = 'ibmq_16_melbourne'

# Can't define the logger up front because we have to define the verbosity level first.
logger = None
PREFERRED_BACKEND = None

""" Driver module for qisquick.  Handles basic startup, configuration, and running experiments.

    Args (globals):
        PREFERRED_PROVIDER_NAME (str): id specifying the provider used by qisquick, if none other is provided.  Defaults
            to 'ibm-q'.  Eligible IDs are found by calling IBMQ.get_providers() and can be passed via the backend param 
            of run_experiment.
        PREFERRED_BACKEND_NAME (str): id specifying the default backend used by qisquick, if none other is provided.
            Defaults to 'ibmq_16_melbourne'.  Eligible backends can be found by calling [provider].backends() and can
            be passed via the backend param of run_experiment.
        logger (Logger): Module-level logger used to determine log message format and define its source.
        PREFERRED_BACKEND (Basebackend): backend object provided for quick access.  Determined by calling 
            [provider].get_backend(PREFERRED_BACKEND_NAME).
"""


def main(argv):
    """ Main function.  Called automatically on execution as main module.

    Args:

    Returns:
        None:
    """

    check_only = argv.check_only
    run_only = argv.run_only
    verbosity = argv.verbose

    _execute(run_local_experiment, check=check_only, run_only=run_only, verbosity=verbosity)


def run_experiment(experiment: Callable, db_location: str = None, provider: str = None,
                   backend: str = PREFERRED_BACKEND_NAME, **kwargs) -> None:
    """ Function provided for run_experiment.py imports into other systems.  This can be called with a user
        defined experiment script and will perform the same tasks as if main() was called on that function

    Args:
        experiment (Callable): The function defined by the user to run their experiment
        db_location (str): Optional.  Should be of the form 'relative/path/dbName.sqlite'
            Defaults to data/circuit_data.sqlite
        provider (str): Optional. String reference used to retrieve provider object.  Defaults to ibm-q
        backend (str): Optional. String reference used to retrieve backend. Defaults to ibmq_16_melbourne
        kwargs: Dictionary of arguments corresponding to those defined by the _get_cli_args function in this module.


    Returns: None.  But as a side-effect will write to the Circs and Stats tables at db_location.
    """
    verbosity = kwargs['verbosity'] if 'verbosity' in kwargs.keys() else None
    _get_logger(verbosity)

    check_only = kwargs['check_only'] if 'check_only' in kwargs.keys() else False
    run_only = kwargs['run_only'] if 'run_only' in kwargs.keys() else False

    _pre_process(provider, backend)
    dbc.set_db_location(db_location)
    dbc.create_all_tables(dbc.db_location)

    _execute(experiment, check_only=check_only, run_only=run_only)


def run_local_experiment() -> List[str]:
    """By Brandon Kamaka, 30 Jan 2020.  Reproducibility experiment to validate test bed
        Create a series of test circuits, and transpile each series with distinct options from various layout and SWAP
        optimizing papers.  Compare success, SWAP efficiency, and time efficiency metrics

    Args:
    Returns:
        List[str]: List of ids of circuits created and tested by this experiment.
    """

    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.passes import LookaheadSwap, DenseLayout

    from qisquick import transpilertools

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
    tests_all_ids = []

    for conf in pass_configurations.keys():
        for circ_case in circuits_to_test.keys():
            test_config = (conf, circ_case)
            logger.info(f'+++++++++++++++++TEST CONFIG: {test_config}++++++++++++++++++++++++++++++++++')

            # ************************** Phase 1: Make initial circuits and prep
            case = circuits_to_test[test_config[1]]
            filename = f'{pass_configurations[test_config[0]]} - {case}'

            # Make the Premades object.
            # It does not contain a circuit but stores the uniform information for circuit creation.
            exp_size = 4
            exp_truth_value = 3
            circ = Premades(size=exp_size, truth_value=exp_truth_value, measure=True)

            # Actually add a specific QuantumCircuit instance based on the stored parameters
            Premades.circ_lib[case](circ)
            circ.draw(output='mpl',
                      filename=filename)
            tests = []
            for i in range(num_trials):
                # Create distinct TestCircuit objects (so that each gets its own unique ID),
                # but each TC gets the same PreMade
                tc = TestCircuit()
                tc.add_circ(circ, size=exp_size, truth_value=exp_truth_value, measure=True)
                tc.stats.name = case
                tc.stats.notes = filename + f' - {i}'
                tests.append(tc)

            # Register initial statistics
            dbc.write_objects(dbc.db_location, tests)

            # ************************** Phase 2, make the distinct PassmMnagers for each test config and circuit

            # Start by getting a transpiler config from the circuits and backend
            level = 1 if pass_configurations[test_config[0]] != 'IBM Optimized' else 3
            configs = transpilertools.get_transpiler_config(circs=tests, be=PREFERRED_BACKEND, optimization_level=level)

            # Then we use the configs to get the appropriate PassManager for each configuration
            pms = []
            for idx, config in enumerate(configs):
                pm = transpilertools.get_basic_pm(config, level=level)
                cm = CouplingMap(PREFERRED_BACKEND.configuration().coupling_map)

                if test_config[1] == 1:
                    pass_type = 'swap'
                    new_pass = LookaheadSwap(coupling_map=cm)

                elif test_config[1] == 2:
                    pass_type = 'layout'
                    new_pass = DenseLayout(coupling_map=cm, backend_prop=PREFERRED_BACKEND.properties())

                else:
                    modified_pm = pm
                    continue

                modified_pm = transpilertools.get_modified_pm(pass_manager=pm, version=level, pass_type=pass_type,
                                                              new_pass=new_pass)
                pms.append(modified_pm)

            # logger.info(f'++++++++++++++++++++++++PM BEING USED++++++++++++++++++++++++++++++++++')
            # logger.info(transpilertools.get_passes_str(pms[0]))
            # logger.info(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            # ************************** Phase 3: Run tests on circuits with custom PassManagers

            # Just in case our number of test_cases exceeds a reasonable size (25)
            test_batches = get_batches(tests)
            pms_batches = get_batches(pms)
            assert len(test_batches) == len(pms_batches)
            for test_batch, pms_batch in zip(test_batches, pms_batches):
                TestCircuit.run_all_tests(test_batch, pass_manager=pms_batch, be=PREFERRED_BACKEND_NAME, attempts=5)

            # ************************** Phase 4: Return final circ ids so the normal routine can save them to Stats
            tests_all_ids.extend([test.id for test in tests])

    return tests_all_ids


def _execute(experiment: Callable, **kwargs):
    """Called either by run_experiment if this library is imported, or by main() if being run as main.
        handles the primary function loop of this application: create/use db, run experiment, check for results
        and register final statistics

    Args:
        experiment (Callable): Experiment to execute.  The provided experiment should accept no arguments and should
            return a list of database IDs to check for circuit execution completion.
        kwargs: Dictionary of arguments corresponding to those defined by the _get_cli_args function in this module.

    Returns:
        None
        """

    global logger

    logger = get_module_logger(__name__)

    run_once = True

    check_only = kwargs['check_only']
    run_only = kwargs['run_only']

    # Prevent duplicates
    ids = set()

    while True:
        if run_once and not check_only:
            new_ids = experiment()
            ids.update(set(new_ids))
            run_once = False

        if run_only:
            break

        else:
            logger.info(f'\t\tCHECKING FOR COMPLETED CIRCUITS...')
            new_ids = check_running()
            ids.update(set(new_ids))
            if dbc.is_empty(dbc.db_location, 'Running'):
                dbc.write_stats(dbc.db_location, ids)
                logger.info(f'All Running jobs have completed.  Shutting down ...')
                break

            # seconds
            sleep(60)

    sys.exit(0)


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


def check_running() -> List[str]:
    """ Checks the "Running" table of the linked db to see if any batches in progress have been executed by the IBM
        backend.  If so, it retrieves the job details, saves them to the main stats db and deletes the record from
        Running.

    Args:

    Returns:
        completed (List[str]): ids of jobs that have completed since last checked.
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

    return list(completed.keys())


def create_all(size: int = 4, truth_value: int = 5, seed: int = None, filename: str = None) -> List[TestCircuit]:
    """ Quick functionality test.  Creates and returns one copy of each test set object, and generates diagrams of them.

    Args:
        size (int): Width of circuit to generate.  Sizes > ~14 qubits are likely to take a long time.
        truth_value (int): Desired base state for the circuit to return under measurement, if executed on an ideal sim.
            This is not applicable for all test circuits (e.g. QFT)
        seed (int): Set random state for reproducibility.
        filename (str): If given, will cause each generated circuit to create a .png of its composer format.

    Returns:
        List[qls.circuits.TestCircuit]:  A list containing one TestCircuit object for each Premades test circuit defined
            in Premades.circ_lib
    """
    circs = []
    for key in Premades.circ_lib.keys():
        new_test = TestCircuit()
        logger.info(f'Creating test circuit for {key} function')
        new_test.add_circ(key, size=size, truth_value=truth_value, measure=True, seed=seed)
        circs.append(new_test)
        print(f'Test circuit ({key}) stats (pre-transpile):\n{str(new_test.stats)}')
        if filename:
            fig = new_test.circuit.draw(output='mpl', filename=f'{filename + "-" + key}.png')

    return circs


def _get_cli_args():
    """ Uses argparse to parse arguments when this module is called directly.  If being imported, these same
        flags can be passed as named parameters to the run_experinment() function call.  Keys are described below:

    Keys:
        check_only (bool): If True the checking routine for recovering executed jobs from the IBM backend is run, but
            the experiment itself is not.  Otherwise both are run
        run_only (bool): Opposite of check_only.  If True the experiment is run but results are not checked for.
            otherwise, both are run
        verbosity (Union[str, int]): Called with -v to -vvv when done from the command line.
            Called with an integer in range(4) if called from import.

    """
    parser = argparse.ArgumentParser(description="driver for testing layout scheme")
    parser.add_argument('--verbose', '-v', action='count', default=None, help='Increase logging level, up to -vvv')
    parser.add_argument('--check-only', '-C', action='store_true',
                        help='If -C, then the program will not run the experiment defined in run_local_experiment(), '
                             'but will only check for job updates for previously submitted jobs.')
    parser.add_argument('--run_only', '-R', action='store_true',
                        help='If -R, then the program will run the experiment defined in run_local_experiment(), '
                             'and will not check for results from the IBM backend.')

    return parser.parse_args()


def _pre_process(default_provider: str = None, default_backend: str = None):
    global PREFERRED_PROVIDER_NAME, PREFERRED_BACKEND_NAME, PREFERRED_BACKEND

    PREFERRED_BACKEND_NAME = default_backend if default_backend is not None else PREFERRED_BACKEND_NAME
    PREFERRED_PROVIDER_NAME = default_provider if default_provider is not None else PREFERRED_PROVIDER_NAME

    # Write global over to TestCircuit
    qqc._preferred_backend = PREFERRED_BACKEND_NAME
    IBMQ.load_account()
    provider = IBMQ.get_provider(PREFERRED_PROVIDER_NAME)
    PREFERRED_BACKEND = provider.get_backend(PREFERRED_BACKEND_NAME)

    return


def _get_logger(verbosity: int) -> None:
    """ Sets up the global logger verbosity parameter and also configures the logger for this module.

    Args:
        verbosity (int): Defines the verbosity level.  Should be in range(4).

    Returns:
        None: But creates logger as a side-effect
    """
    global logger

    config_logger(verbosity)
    logger = get_module_logger(__name__)
    logger.critical('Showing critical messages')
    logger.error('Showing error messages')
    logger.warning('Showing warning messages')
    logger.info('Showing informational messages')
    logger.debug('Showing debug messages')


if __name__ == '__main__':
    _pre_process()
    args = _get_cli_args()
    _get_logger(args.verbose)
    main(args)

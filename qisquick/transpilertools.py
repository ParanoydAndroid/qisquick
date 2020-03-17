from typing import List, Union

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler.transpile import _parse_transpile_args
from qiskit.providers import basebackend
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.preset_passmanagers import *
from qiskit.transpiler.transpile_config import TranspileConfig

from qisquick.circuits import TestCircuit
from qisquick.run_experiment import PREFERRED_BACKEND

"""
Module of unbound helper functions to integrate into the Qiskit transpiler workflow.

"""


def get_transpiler_config(circs: Union[List[TestCircuit], TestCircuit, List[QuantumCircuit], QuantumCircuit],
                          be: basebackend = PREFERRED_BACKEND, layout: Layout = None, optimization_level: int = None,
                          callback: callable = None) -> List[TranspileConfig]:
    """ Given a list of circuits and a backend to execute them on, return a list of transpiler configs of the same
        length such that configs[i] is the config for circs[i]

    Args:
        circs (Union[List[TestCircuit], TestCircuit]): Single circuit or List of circuits to
            compile configurations for
        be (IBMQBackend): Backend object to execute the circuits on.
        layout (Layout): Optional.  Initial layout to use.
        optimization_level (int): Optional. IBM transpiler optimization level to target [0, 3].
        callback (Callable): Optional. Function to call at the end of execution of each pass in the PassManager.

    Returns:
        List[TranspileConfig]: List of transpiler configurations associated with
            circs.  Also writes to self.transpiler_config as a side-effect.
    """
    # First, parse the input type of circs and process it correctly to return a list of only QuantumCircuits
    # Also set a flag to save TestCircuit.transpiler_config to member field if TestCircuits were provided.
    circuits = []
    save_configs = False
    if isinstance(circs, List):
        if isinstance(circs[0], TestCircuit):
            circuits = [tc.circuit for tc in circs]
            save_configs = True
        elif isinstance(circs[0], QuantumCircuit):
            circuits = circs
    elif isinstance(circs, TestCircuit):
        circuits.append(circs.circuit)
        circs = [circs]
        save_configs = True
    elif isinstance(circs, QuantumCircuit):
        circuits.append(circs)
    else:
        raise TypeError(f'The circuit must be a single QuantumCircuit (or subclass) or list of elements of that type.  '
                        f'Instead received: {type(circs)}')

    # _parse_transpile_args will call _parse_x_args() where x is each parameter type.
    # If this parameter is None, then each _parse_x_arg function will retrieve that parameter from backend.
    # Hence backend being the only requirement.  All other params exposed by get_transpiler_config are for custom tests
    configs = _parse_transpile_args(circuits, backend=be, basis_gates=None, coupling_map=None,
                                    backend_properties=None,
                                    initial_layout=layout, seed_transpiler=None,
                                    optimization_level=optimization_level,
                                    pass_manager=None, callback=callback, output_name=None)

    if save_configs:
        for idx, circ in enumerate(circs):
            circ.transpiler_config = configs[idx]

    return configs


def get_basic_pm(transpiler_config: TranspileConfig, level: int = 0) -> PassManager:
    """ Get a pre-populated PassManager from the native Qiskit implementation.

    Args:
        transpiler_config (TranspileConfig): Configuration used to generate the
            tailored PassManager.
        level (int): Optional. Qiskit Transpiler optimization level to target.

    Returns:
        PassManager: PassManager instance associated with the provided config.
    """

    # TODO Add pm to stats and db
    pm_funcs = {
        0: level_0_pass_manager,
        1: level_1_pass_manager,
        2: level_2_pass_manager,
        3: level_3_pass_manager
    }

    pm_func = pm_funcs[level]
    return pm_func(transpiler_config)


def get_modified_pm(pass_manager: PassManager, version: int, pass_type: str, new_pass: BasePass) -> PassManager:
    """ Modifies a provided PassManager instance by exchanging swap or layout passes with others of the same basic type.

    Args:
        pass_manager (qiskit.transpiler.passmanager.PassManager): PassManager instance to modify.
        version (int): Which optimization level the original PassManager was targeted at.
        pass_type (str): Type of pass to exchange.  Must be one of ('swap', 'layout')
        new_pass (BasePass): The pass to insert into pass_manager in place of that pass_manager's pass of type (type).

    Returns:
        qiskit.transpiler.passmanager.PassManager: Modified PassManager instance.
    """

    # TODO add pms to stats and db

    if version not in range(4):
        raise ValueError(f'version must correspond to an existing optimization level (range(4)).  Got {version}')

    if not pass_type == 'swap' and not pass_type == 'layout':
        raise KeyError(f'Can only exchange swap or layout passes.  Was given type {pass_type}')

    # Map of which indices the pass of each type are located at for each basic pm
    locations = {
        0: {'swap': (6, 0), 'layout': (1, None)},
        1: {'swap': (7, 1), 'layout': (1, None)},
        2: {'swap': (6, 1), 'layout': (2, None)},
        3: {'swap': (6, 1), 'layout': (2, None)}
    }

    # The particular pass to replace depends on which test group the given pm is supposed to work for

    # The transpiler passmanager format is a mess.  The PassManager is actually gives us a list of dictionaries
    # of dictionaries of lists of passes.  No, I'm not kidding.
    pass_list = pass_manager.passes()
    first_index = locations[version][pass_type][0]
    inner_index = locations[version][pass_type][1]
    passes_dict = pass_list[first_index]

    passes = []
    for i in range(len(passes_dict['passes'])):
        if inner_index is None:
            passes = new_pass
            break
        elif i == inner_index:
            passes.append(new_pass)
        else:
            passes.append(passes_dict['passes'][i])

    pass_manager.replace(first_index, passes)

    return pass_manager


def get_passes_str(pm: PassManager) -> str:
    """ Returns str of actual passes embedded in the provided PassManager object."""
    pass_list = pm.passes()

    msg = ''
    for idx, pass_dict in enumerate(pass_list):
        for actual_goddamn_pass in pass_dict['passes']:
            msg += f'{idx}: {actual_goddamn_pass.name()}\n'

    return msg


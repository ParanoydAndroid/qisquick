# QISquick

Helper library for automating experimental setup, configuration, and data generation of Qiskit transpiler functionality.

qisquick provides a test set of defined quantum circuits, functions to help modify transpiler configuration and passmanagers, and a database system for saving experimental data including backend state, transpilation statistics, and execution statistics.

## Using qisquick

qisquick can be thought of as composed of two use-cases: experiment scripting and experiment execution.  A quick use guide is provided here,
but more information can be found in the documentation.pdf file of this package, which includes documentation on individual modules,
classes, and functions.

### Experiment Scripting

Researchers must provide an experiment defined as a Callable that takes no arguments and returns circuit ids.  A minimal example:

    def my_experiment():
        # Creates TestCircuit object and automatically generates a Premades circuit of type 'two_bell'
        circ = TestCircuit.generate(case='two_bell', size=4) 
        
        return [circ.id]

qisquick is in particular designed to ease experimenting with various transpiler configurations, and so many tools are provided to make scripting such experiments easier.  A more comprehensive coverage of the purpose of the various classes can be found in the actual system documentation by calling help() on any function, but a few examples are provided here.

The Premades class provides ~10 circuits designed to demonstrate various constraint topologies, be representative subcircuits, interesting primitives, or full algorithm implementations.  All circuits can be generated using the same common interface, by providing a size (circuit width), and a truth value: a base state that should be encoded into oracles and used if the circuit requires a 'correct answer' (e.g. Grover's search).  

The TranspilerTools class provides helper methods to take qiskit-defined pre-populated PassManagers and exchange SWAP or layout passes with those defined by the researcher.

The TestCircuit class allows researchers to easily generate statistics about TestCircuits, including transpile time, SWAPS inserted by transpiler routines, ideal distributions, and actual execution statistics.

An experiment showing these behaviors:

    from qisquick.run_experiment import *
    from qisquick.circuits import Premades, TestCircuit
    from qisquick.transpilertools import get_transpiler_config, get_basic_pm, get_modified_pm

    def my_experiment() -> list:
        # Make one of each kind of Premade and attach each to a TestCircuit
        size = 4
        tv = 3
        circs = create_all(size=size, truth_value=tv)
    
        # We could also generate the circs with a seed so they are reliably recreated
        # and also pass a filename to create image output of the circuits
        # circs = create_all(size=4, truth_value=3, seed = 404, filename='prepend_to_circs')
    
        # We could also add a specific type of circuit, defined in Premades.circ_lib
        new_test_circ = TestCircuit.generate(case='two_bell', size=size, truth_value=tv)
        circs.append(new_test_circ)
    
        # Or we can make an empty TestCircuit and add our own custom circ (or an existing Premades)
        tc = TestCircuit()
        custom_circ = Premades(size=size, truth_value=tv)
        custom_circ.h(0)
        custom_circ.cx(0, 1)
        custom_circ.measure(custom_circ.qr, custom_circ.cr)

        # You have to tell the TestCircuit what the size and tv params of the Premades is; should be fixed in the future
        tc.add_circ(custom_circ, size=size, truth_value=tv)
        circs.append(tc)
    
        # Now let's save a transpiler configuration for each circuit.  
        provider = IBMQ.load_account()
        backend = provider.get_backend('MY PREFERRED BACKEND')
        
        # We must have a backend to target.  This can be provided now, or at time of experiment execution.
        # Note that get_transpiler_config actually writes to each TestCircuit's transpiler_config param.
        # the configs returned are just for convenience if desired.
        configs = transpilertools.get_transpiler_config(circs)
    
        # Assume we have a new Passmanager SWAP pass we'd like to test out.  This is provided by the researcher.
        my_swap = get_my_pass()
    
        # We generate a basic, optimization_level 3 PassManager for each of our TestCircuits, then exchange in our new pass
        pms = []
        for circ in circs:
            pm = get_basic_pm(circ.transpiler_config, level=3)
            modified_pm = transpilertools.get_modified_pm(pass_manager=pm, version=3, pass_type='swap', new_pass=my_swap)
            pms.append(modified_pm)
        
        # We can also execute experiments on our new pass
        # This tests transpilation time (average over attempts trials), generates compiled circs using the pm and stores
        # the compiled_circs attribute, and submits the job for execution on the backend chosen at runtime.
        TestCircuit.run_all_tests(circs, pms, attempts=5)
        
        # We return these ids of our Testcircuits
        return [circ.id for circ in circs]

### Experiment Execution

Once the researcher has an experiment they'd like to run, this can be handled very easily through one of two interfaces:

* For researchers who may want to modify actual qisquick functionality, or just prefer it, the experimental code can be inserted into the `run_local_experiment()` method of `run_experiment.py`.  This module can then be called via normal CLI calls and will execute that experiment.  Additionally, researchers can pass 3 flags (documentation also available by calling run_experiment.py -h):

    -v (-vv, -vvv): Increase the verbosity level of the logging system.
    
    -C, --Check_only: Checks only for previously run experimental results from IBM.  Executes no experiment script.
    
    -R, --run_only: Runs experiment, but does not check for results from IBM.  Runs local tests and, if called, send circuits for execution but quits immediately thereafter.

* Alternatively, in any module the researcher can `from qisquick.run_experiment import run_experiment` and can call this function to execute their experiment script.  This function also accepts arguments equivalent to the previously mentioned flags and additionally allows the researcher to define their default backend, database location, etc ...

     Since the functions we used as examples when writing our script are built to use whatever default backend is called with run_experiment(), then this means that a researcher *can* define a backend inside their script, but if they don't do so, then they can do it on calling the experiment and have their new default replicated automatically throughout their script.
     
     Instead of the CLI flags, this function accepts named params with the following keys: verbosity (int, range(4)), check\_only (bool), run\_only (bool).    
     
        my_module.py
        from qisquick.run_experiment import run_experiment
        
        # verbosity = '-vv' (no flag is verbosity level 1)
        run_experiment(my_experiment, provider=my_provider, backend=my_backend, verbosity=3)
        
    If no backend and provider are provided, the library defaults are the public ibm-q provider and the public melbourne backend.  Researchers can also pass a custom dblocation -- to create the db or reference an existing sqlite3 db of the correct structure.  If no dblocation is provided, one is created at data/circuit_data.sqlite  
 




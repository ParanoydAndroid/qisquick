import uuid


class Statblock:
    """ Associated with each TestCircuit on creation.  Stores object properties relevant for analyzing experimental
        results or reproducing them."""

    def __init__(self, parent):
        self.id = uuid.uuid4().hex
        self.parent = parent
        self.name = None
        self.truth_value = None
        self.ideal_distribution = None
        self.results = None

        self.circ_width = None
        self.pre_depth = None
        self.seed = None

        self.backend = None
        self.post_depth = None
        self.swap_count = None

        self.compile_time = None

        self.datetime = None
        self.iteration = None

        self.batch_avg = None
        self.global_avg = None

        self.notes = None

    @property
    def job_id(self):
        return self.parent.job_id

    @job_id.setter
    def job_id(self, new_id: str):
        self.parent.job_id = new_id

    @property
    def backend(self):
        return self.parent.backend

    @backend.setter
    def backend(self, new_backend):
        self.parent.backend = new_backend

    def __str__(self):
        fmt = '{}: {}\n'
        result = ''
        for key in self.to_dict().keys():
            result += fmt.format(key, self.to_dict()[key])
        return result

    def to_dict(self, writeable: bool = False):
        from copy import deepcopy

        # Default to a deepcopy so it can't accidentally get modified
        if writeable:
            return self.__dict__
        else:
            return deepcopy(self.__dict__)

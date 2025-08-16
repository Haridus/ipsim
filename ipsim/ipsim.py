#================================================================
class ValueSource(object):
    """Class representing value source - object that can return constant values 
       or request external source for value """
    
    def __init__(self,*,value = 0, source = None):
        """Constructor
        value - constan value that returned when source is None
        source - external source that return value, must can be called by () operator
        """
        self._value = value
        self._source = source

    def value(self):
        """return underlying value or value from underlying source"""
        if(self._source != None):
            return self._source()
        return self._value
    
    def __call__(self):
        """sinonim for get_value to support () call convention"""
        return self.value()
    
    def __add__(self, value):
        """Reimplement + operator. Will only work if value type returned by value() also supports + operator"""
        return self.value()+value

    def __sub__(self, value):
        """Reimplement - operator. Will only work if value type returned by value() also supports - operator"""
        return self.value()-value
    
    def __mul__(self, value):
        """Reimplement * operator. Will only work if value type returned by value() also supports * operator"""
        return self.value()*value
    
    def __truediv__(self, value):
        """Reimplement / operator. Will only work if value type returned by value() also supports / operator"""
        return self.value() / value

class ProcessVariable(ValueSource):
    """Class that represents process variable
    it ValueSource that allow change source of values
    
    TODO:
    add metadata such as measurement units, constraints etc.
    """
    def __init__(self, *, value = 0, source = None):
        super().__init__(value=value, source=source)
        
    def set_source(self, source):
        self._source = source

#----------------------------------------------------------------
class VariablesCategory:
    """Helper class to allow specifi input/output pins and states as static class fields
    """
    def __init__(self, variables=None):
        super().__init__()
        self.variables = variables

class Inputs(VariablesCategory):
    """Helper class to allow specifi input pins as static class fields
    """
    def __init__(self, variables=None):
        super().__init__(variables)

class Outputs(VariablesCategory):
    """Helper class to allow specifi output pins as static class fields
    """
    def __init__(self, variables=None):
        super().__init__(variables)

class States(VariablesCategory):
    """Helper class to allow specifi states varaiables as static class fields
    """
    def __init__(self, variables=None):
        super().__init__(variables)

class Disturbances(VariablesCategory):
    """Helper class to allow specifi desturbance varaiables as static class fields
    """
    def __init__(self, variables=None):
        super().__init__(variables)

class _Internal:
    """Helper class to store variables by categories as named entyties
    """
    def __init__(self):
        pass

class NodeEvaluationResult:
    """Helper class that evaluates node on request
    calling internal _evaluate() method that use cached results on time t
    allowing to not reevaluate node value for several target nodes
    """
    def __init__(self, node, name):
        self._node = node
        self._name = name

    def __call__(self):
        self._node._evaluate()
        return self._node._result(self._name)

class ProcessNode(object):
    """Main node class that allov evaluate node value for inputs on time t
    and exports self value sources for bound target nodes 
    use internal caching of evaluation results on time t to calculate result only once

    """
    
    @classmethod
    def _collect_fields(cls):
        """Collect all Field instances from class attributes"""

        inputs  = None
        outputs = None
        states  = None
        disturbances = None

        for name, value in cls.__dict__.items():
            if not name.startswith('_'):
                if isinstance(value, Inputs):
                    inputs = value
                elif isinstance(value, Outputs):
                    outputs = value
                elif isinstance(value, States):
                    states = value
                elif isinstance(value, Disturbances):
                    disturbances = value

        return inputs, outputs, states, disturbances

    def __init__(self, name):
        self._name = name
        self._inputs = {}
        self._outputs = {}
        self._results = {}
        self._model = None
        self._current_time = 0
        self._u = _Internal()
        self._x = _Internal()
        self._o = _Internal()
        self._e = _Internal()
        
        
        inputs, outputs, states, disturbances = self._collect_fields()
        
        if inputs is not None:
            self.create_inputs(inputs.variables)
                
        if outputs is not None:
            self.create_outputs(outputs.variables)
       
        if states is not None:
            for variable_name in states.variables:
                self._x.__dict__[variable_name] = object()

        if disturbances is not None:
            for variable_name in disturbances.variables:
                self._e.__dict__[variable_name] = object()

    def name(self):
        """Return name of the node"""
        return self._name
            
    def create_input(self, name):
        """Create input variable source"""
        self._inputs[name] = ProcessVariable()
        self._u.__dict__[name] = self._inputs[name]

    def create_inputs(self, names):
        """Create inputs variables sources"""
        for _ in names:
            self.create_input(_)
        
    def set_input_source(self, name, source):
        """Sets source of input variable"""
        self._inputs[name].set_source(source)
    
    def inputs(self):
        """Return input variables of the node. Ment to be used in evaluation method"""
        return self._inputs
    
    def set_result(self, name, value):
        """Caches node evaluation result. Ment to be used in evaluation method"""
        self._results[name] = value

    def _result(self, name):
        """Get cached result by name. Ment to be used in NodeEvaluationResult helper"""
        return self._results[name]

    def _evaluate(self):
        """Internal evaluation method that work on evaluation cache"""
        if (self._model is None) or (self._model.time() != self._current_time):
            self._results.clear()

        if not self._results:
            self.evaluate()
            self._current_time = self._model.time()
                        
    def evaluate(self):
        """Evaluation method that must be reimplemented in childs.
           Default behaviour set all outputs to 0
        """
        for key in self._outputs:
            self.set_result(key, 0)  

    def create_output(self, name):
        """Create output variable source"""
        self._outputs[name] = ProcessVariable(source=NodeEvaluationResult(self,name))
        self._o.__dict__[name] = self._outputs[name]

    def create_outputs(self, names):
        """Create outputs variables sources"""
        for _ in names:
            self.create_output(_)

    def outputs(self):
        return self._outputs
    
    def value(self, name):
        return self._result(name)

    def state_value(self, name):
        return self._x.__dict__[name]
    
    def change_state_value(self, name, value):
        if name in self._x.__dict__:
            self._x.__dict__[name] = value

class ProcessInputNode(ProcessNode):
    """Input node class 
    Input nodes does have other bound sources exept output sources 
    Input nodes only provides plain values for outputs
    """

    def __init__(self, name, parameters):
        """parameters is a dict-like object that specifies key-value pairs
          key is the name of the output
          value is the value of the output
        """ 
        super().__init__(name)
        for key in parameters:
            self.create_output(key)
        self._results = parameters
        
    def _evaluate(self):
        """dont evaluate -> we just return stored values to outputs"""
        pass
    
    def value(self, name):
        return self._result(name)

    def state_value(self, name):
        return self.value(self,name)

    def change_state_value(self, name, value):
        self.set_result(name,value)

class Sensor(ProcessNode):
    """Output node
    Sensor operates on single output field of other node basicaly copying inputs to outputs
    But evaluation method can be reimplemented for example to add noise to output 
    """
    def __init__(self, name, input_name):
        super().__init__(name)
        self.create_input(input_name)
        self.create_output(input_name)

    def evaluate(self):
        i = self.inputs()
        for key in i:
            self.set_result(key, i[key]())

#----------------------------------------------------------------
class ProcessModel(object):
    """Process model. Basicaly manager of nodes"""

    class Time:
        """Class that represents model time. Total time and time increment dt
        """
        def __init__(self, *, start_time = 0, dt=0.01):
            """Constructor specifying start time and time increment"""
            self._time = start_time
            self._dt = dt

        def next(self):
            """generate next time"""
            self._time = self._time + self._dt
            return self._time

        def time(self):
            """return current time"""
            return self._time
        
        def dt(self):
            """"return time increment"""
            return self._dt

    def __init__(self, name = "", *, dt = 0.01, manipulator = None, observer = None):
        """constructor"""
        self._name = name
        self._time = self.Time(start_time=-dt, dt=dt) #set start time to -dt to evaluate on time 0
        self._nodes = {}
        self._manipulator=manipulator
        self._observer=observer
    
    def time(self):
        """return current model time"""
        return self._time.time()
    
    def dt(self):
        """return current model time increment"""
        return self._time.dt()

    def nodes(self):
        """return current model nodes"""
        return self._nodes

    def add_node(self, node):
        """add node to model"""
        node._model = self
        self._nodes[node.name()] = node

    def bond_nodes(self, dest_node, dest_node_field, source_node, source_node_field):
        """bound nodes by pins
        """
        dn = self._nodes[dest_node]
        sn = self._nodes[source_node]
        dn.set_input_source(dest_node_field,sn.outputs()[source_node_field])

    def bond_pins(self, dest_pin, source_pin):
        """bound nodes directly by pins
           Pins must be obtained by addressing variables from _u and _o pins aggregators of ProcessNode class
          """
        dest_pin.set_source(source_pin)

    def _evaluate(self, nodes_names):
        result = {"_model_time_":self.time()}
        for node_name in nodes_names:
            node = self._nodes[node_name]
            noutputs = node.outputs()
            result[node.name()] = { key: noutputs[key]() for key in noutputs }
        return result
    
    def next_state(self, *, nodes_names = None, action = None):
        """synonym for step"""
        return self.step(nodes_names=nodes_names, action=action)
    
    def step(self, *, nodes_names = None, action = None):
        """generate next model state. Triger reevaluation of named nodes and all its source nodes"""
        if (action is not None) and (self._manipulator is not None):
            self._manipulator(self, action)

        self._time.next()
        if nodes_names is None:
            nodes_names = self._nodes.keys()

        _state = None
        if self._observer is not None:
            _state = self._observer(self, self._evaluate(self._nodes.keys()))
        else:
           _state = self._evaluate(nodes_names)

        return _state
    
    def reset_time(self, dt):
        """resets model time"""
        #TODO: reset also nodes states if needed
        self._time = self.Time(start_time=-dt, dt=dt)

    @staticmethod
    def make_common_manipulator(affecting_pins_list):
        def __common_manipulator(model, action):
            nodes = model.nodes()
            for _ in range(min(len(affecting_pins_list), len(action))):
                node_name, pin_name = affecting_pins_list[_]
                nodes[node_name].change_state_value(pin_name, action[_])
        return __common_manipulator
    
    @staticmethod
    def make_common_objerver(affecting_pins_list):
        def __common_observer(model, state):
            nodes = model.nodes()
            result = []
            for _ in range(len(affecting_pins_list)):
                node_name, pin_name = affecting_pins_list[_]
                result.append(nodes[node_name].value(pin_name))
            return result
        return __common_observer
        
#================================================================
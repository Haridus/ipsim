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
        """return some value"""
        if(self._source != None):
            return self._source()
        return self._value
    
    def __call__(self):
        """sinonim for get_value to support () call convention"""
        return self.value()

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
    
    TODO:
    allow child classes to specifi input/output valiable sources as static metadata fields
    """
    
    def __init__(self, name):
        self._name = name
        self._inputs = {}
        self._outputs = {}
        self._results = {}
        self._model = None
        self._moment = 0

    def name(self):
        """Return name of the node"""
        return self._name
    
    def create_input(self, name):
        """Create input variable source (will be removed in future versions)"""
        self._inputs[name] = ProcessVariable()

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
        if (self._model is None) or (self._model.time() != self._moment):
            self._results.clear()

        if not self._results:
            self.evaluate()
            self._moment = self._model.time()
                        
    def evaluate(self):
        """Evaluation method that must be reimplemented in childs.
           Default behaviour set all outputs to 0
        """
        for key in self._outputs:
            self.set_result(key, 0)  

    def create_output(self, name):
        """Create output variable source (will be removed in future versions)"""
        self._outputs[name] = ProcessVariable(source=NodeEvaluationResult(self,name))

    def outputs(self):
        return self._outputs

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

    def change_value(self, name, value):
        self.set_result(name,value)

class Sensor(ProcessNode):
    """Output node
    Sensor operates on single output field of other node basicaly copying inputs to outputs
    But evaluation method can be reimplemented for example to add noise to output 
    """
    def __init__(self, name, inputName):
        super().__init__(name)
        self.create_input(inputName)
        self.create_output(inputName)

    def evaluate(self):
        i = self.inputs()
        for key in i:
            self.set_result(key, i[key]())

#----------------------------------------------------------------
class ProcessModel(object):
    """Process model. Basicaly manages of nodes"""

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

    def __init__(self, name = "", *, dt = 0.01):
        """i"""
        self._name = name
        self._time = self.Time(start_time=-dt, dt=dt) #set start time to -dt to evaluate on time 0
        self._nodes = {}
    
    def time(self):
        return self._time.time()
    
    def dt(self):
        return self._time.dt()

    def nodes(self):
        return self._nodes

    def add_node(self, node):
        node._model = self
        self._nodes[node.name()] = node

    def bond_nodes(self, dest_node, dest_node_field, source_node, source_node_field):
        dn = self._nodes[dest_node]
        sn = self._nodes[source_node]
        dn.set_input_source(dest_node_field,sn.outputs()[source_node_field])

    def _evaluate(self, nodes_names):
        result = {"_model_time_":self.time()}
        for node_name in nodes_names:
            node = self._nodes[node_name]
            noutputs = node.outputs()
            result[node.name()] = { key: noutputs[key]() for key in noutputs }
        return result

    def next_state(self, nodes_names):
        """generate next model state. Triger reevaluation of named nodes and all its source nodes"""
        self._time.next()
        return self._evaluate(nodes_names)
    
    def reset_time(self, dt):
        """resets model time"""
        #TODO: reset also nodes states if needed
        self._time = self.Time(start_time=-dt, dt=dt)
    
#================================================================
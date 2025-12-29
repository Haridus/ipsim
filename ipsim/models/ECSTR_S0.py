from ipsim import *
from numpy import exp, seterr, pi

#====================================================================================
class ECSTR_S0_Node(ProcessNode):
    I  = Inputs(  ("Tc", "q_out", ) )
    O  = Outputs( ("cA", "T", "h",) )
    U  = States(  ("cA", "T", "h",) )

    class Config:
        def __init__(self):
            self.q_in = .1  # m^3/min
            self.Tf = 76.85  # degrees C
            self.cAf = 1.0  # kmol/m^3
            self.r = .219  # m
            self.k0 = 7.2e10  # min^-1
            self.E_divided_by_R = 8750  # K
            self.U = 54.94  # kg/min/m^2/K
            self.rho = 1000  # kg/m^3
            self.Cp = .239  # kJ/kg/K
            self.dH = -5e4  # kJ/kmol
            self.T_base = 273.15

    def __init__(self, name, *
                , config = Config(), solver = None
                , cA = 0.8778252, T = 51.34660837, h = 0.659):
        super().__init__(name)
        self._x.cA = cA
        self._x.T  = T
        self._x.h = h
        self.config = config
        self.solver = solver
        
    def ode(self, x, u, c):
        cA, T, h = x
        Tc, q_out = u
        rate = c.k0 * cA * exp(-c.E_divided_by_R / (T + c.T_base))  # kmol/m^3/min

        return [
            c.q_in * (c.cAf - cA) / (pi * c.r ** 2 * h + 1e-8) - rate,  # kmol/m^3/min
            c.q_in * (c.Tf - T) / (pi * c.r ** 2 * h + 1e-8)
            - c.dH / (c.rho * c.Cp) * rate
            + 2 * c.U / (c.r * c.rho * c.Cp) * (Tc - T),  # degree C/min
            (c.q_in - q_out) / (pi * c.r ** 2)  # m/min
        ]
    
    def evaluate(self):   
        soln = self.solver( lambda t, x, u, c: self.ode(x, u, c)
                          , [0, self._model.dt()]
                          , [self._x.cA,self._x.T, self._x.h,]
                          , (self._u.Tc(), self._u.q_out(), )
                          , self.config )
        
        self._x.cA = soln.y[0, :][-1]
        self._x.T  = soln.y[1, :][-1]
        self._x.h  = soln.y[2, :][-1]

        self.set_result("cA", self._x.cA)
        self.set_result("T", self._x.T)
        self.set_result("h", self._x.h)
                
class ECSTR_S0(ProcessModel):
    def __init__(self, solver, *
                , dt = 0.1
                , init_state = None
                , observer = None
                , manipulator = None):
        
        super().__init__("ECSTR_S0_", dt = dt, observer = observer, manipulator = manipulator)
        coolant           = ProcessInputNode("Coolant", {"T":26.85})
        out_flow_control  = ProcessInputNode("OutFlowControl", {"q":0.1})
        
        reactor   = ECSTR_S0_Node("Reactor", solver=solver) if (init_state is None) else ECSTR_S0_Node("Reactor",cA=init_state["cA"], T=init_state["T"], h=init_state["h"], solver=solver)

        sensorA = Sensor("SensorA","cA")
        sensorT = Sensor("SensorT","T")
        sensorH = Sensor("SensorH","h")
        
        self.add_node(coolant)
        self.add_node(out_flow_control)
        self.add_node(reactor)
        self.add_node(sensorA)
        self.add_node(sensorT)
        self.add_node(sensorH)

        self.bond_nodes("Reactor","Tc","Coolant","T")
        self.bond_nodes("Reactor","q_out","OutFlowControl","q")

        self.bond_nodes("SensorA","cA", "Reactor", "cA")
        self.bond_nodes("SensorT","T", "Reactor", "T")
        self.bond_nodes("SensorH","h", "Reactor", "h")
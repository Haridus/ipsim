import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../ipsim")) 

from ipsim import *
from numpy import exp 

from numpy import empty 
from scipy.integrate import solve_ivp

#==========================================================
class CSTRSimpleReactorNode(ProcessNode):
    I  = Inputs(  ("qi", "cAi", "Ti", "Tc") )
    O  = Outputs( ("cA", "cB", "T",) )
    U  = States(  ("cA", "cB", "T",) )

    class Config:
        def __init__(self):
            self.Ea  = 72750     # activation energy J/gmol
            self.R   = 8.314     # gas constant J/gmol/K
            self.k0  = 7.2e10    # Arrhenius rate constant 1/min
            self.dHr = -5.0e4    # Enthalpy of reaction [J/mol]
            self.rho = 1000.0    # Density [g/L]
            self.Cp  = 0.239     # Heat capacity [J/g/K]
            self.UA  = 5.0e4     # Heat transfer [J/min/K]
            self.V   = 100.0     # Volume [L]

    def __init__(self, name, *, cA = 0, cB = 0, T = 298.15, config = Config(), solver = None):
        super().__init__(name)
        self._x.cA = cA
        self._x.cB = cB
        self._x.T  = T
        self.config = config
        self.solver = solver
        
    def ode(self, x, u, c):
        cA, cB, T,  = x
        qi, cAi, Ti, Tc = u

        qV    = qi/c.V
        kCa   = cA*c.k0*exp(-c.Ea/c.R/T)
        dHpC  = -c.dHr/(c.rho*c.Cp)
        UAVpC = c.UA/(c.V*c.rho*c.Cp)
        
        dcAdt = qV*(cAi - cA) - kCa
        dTdt  = qV*(Ti - T) + dHpC*kCa + UAVpC*(Tc-T)
        dBdt  = qV*(0 - cB) + kCa

        return [dcAdt, dBdt, dTdt]
    
    def evaluate(self):   
        self.set_result("cA",self._x.cA)
        self.set_result("cB",self._x.cB)
        self.set_result("T",self._x.T)

        soln = self.solver(lambda t, x, u, c: self.ode(x, u, c)
                        , [0, self._model.dt()]
                        , [self._x.cA,self._x.cB, self._x.T]
                        , args = ( (self._u.qi(), self._u.cAi(), self._u.Ti(), self._u.Tc() )
                                 , self.config) )
        
        self._x.cA = soln.y[0, :][-1]
        self._x.cB = soln.y[1, :][-1]
        self._x.T  = soln.y[2, :][-1]
        
class CSTRSimple(ProcessModel):
    def __init__(self, solver, *
                , dt = 10.0/2000, init_state = None , observer = None):
        if observer is None:
             observer = ProcessModel.make_common_objerver([ ("SensorA", "cA")
                                                          , ("SensorB", "cB")
                                                          , ("SensorT", "T") ])
        super().__init__("CSTRSimple", dt = dt, observer = observer)

        inlet = ProcessInputNode("InletFeed", {"Flowrate":100,"cA":1,"T":350})
        coolant   = ProcessInputNode("Coolant", {"T":315})
        reactor   = CSTRSimpleReactorNode("Reactor", solver=solver) if (init_state is None) else CSTRSimpleReactorNode("Reactor",cA=init_state["cA"], cB=init_state["cB"], T=init_state["T"], solver=solver)

        sensorA = Sensor("SensorA","cA")
        sensorB = Sensor("SensorB","cB")
        sensorT = Sensor("SensorT","T")
        
        self.add_node(inlet)
        self.add_node(coolant)
        self.add_node(reactor)
        self.add_node(sensorA)
        self.add_node(sensorB)
        self.add_node(sensorT)
    
        self.bond_nodes("Reactor","qi","InletFeed","Flowrate")
        self.bond_nodes("Reactor","cAi","InletFeed","cA")
        self.bond_nodes("Reactor","Ti","InletFeed","T")
        self.bond_nodes("Reactor","Tc","Coolant","T")

        self.bond_nodes("SensorA","cA", "Reactor", "cA")
        self.bond_nodes("SensorB","cB", "Reactor", "cB")
        self.bond_nodes("SensorT","T", "Reactor", "T")

#==========================================================================================

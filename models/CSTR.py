import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../ipsim")) 

from ipsim import *
from numpy import exp 

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

#--------------------------------------------------------------------
class KlattEngellReactorNode(ProcessNode):
    """
    See model description at 10.1016/j.jprocont.2006.09.002
    """
    I  = Inputs(  ("F", "Q",) )
    O  = Outputs( ("F", "Tc",) )
    U  = States(  ("F", "Tc",) )

    class IFlow:
        def __init__(self, cA, T, q):
            self.cA = cA
            self.T  = T
            self.q  = q

    class OFlow:
        def __init__(self, cA, cB, cC, cD, T):
            self.cA = cA
            self.cB = cB
            self.cC = cC
            self.cD = cD
            self.T  = T

    class Config:
        def __init__(self):
            self.alpha = 30.8285 #h-1
            self.gamma = 0.1 # K/kJ
            self.betta = 86.688 #h-1
            self.delta = 3.556*10e-4 # m3*K/kJ
            
            self.k_0_AB = 1.287*1e12 # h-1
            self.k_0_BC = 1.287*1e12 # h-1
            self.k_0_AD = 9.043*1e6  # m3/mol/h
    
            self.dH_R_AB = 4.2       # kJ/molA
            self.dH_R_BC = -11.0     # kJ/molB
            self.dH_R_AD = -41.85    # kJ/molA
    
            self.E_A_AB = 9758.3  # The activation energy parameters Ei already comprise the gas constant and can therefore be noted without units.
            self.E_A_BC = 9758.3  # The activation energy parameters Ei already comprise the gas constant and can therefore be noted without units.
            self.E_A_AD = 8560.0  # The activation energy parameters Ei already comprise the gas constant and can therefore be noted without units.

    def __init__(self, name, *
                , cA = 3517.5
                , cB = 740
                , cC = 0
                , cD = 0
                , T = 360.15
                , Tc = 352.15
                , config = Config()
                , solver = None):
        super().__init__(name)
        self.config = config
        self.solver = solver

        self._x.F   = self.OFlow(cA, cB, cC, cD, T)
        self._x.Tc  = Tc # K        

    def ode(self, x, u, c):
        cA, cB, cC, cD, T, Tc = x
        Fi, Q = u

        k_A_AB = c.k_0_AB*exp(-c.E_A_AB/T)
        k_A_AD = c.k_0_AD*exp(-c.E_A_AD/T)
        k_A_BC = c.k_0_BC*exp(-c.E_A_BC/T)

        dcAdt = Fi.q*(Fi.cA - cA) - cA*k_A_AB- cA**2*k_A_AD
        dcBdt = -Fi.q*cB + cA*k_A_AB - cB*k_A_BC
        dcCdt = -Fi.q*cC + cB*k_A_BC
        dcDdt = -Fi.q*cD + cA**2*k_A_AD

        dTdt  = c.delta*( cA*k_A_AB*c.dH_R_AB \
                        + cB*k_A_BC*c.dH_R_BC  \
                        + cA**2*k_A_AD*c.dH_R_AD) \
                        + c.alpha*(Tc - T) + Fi.q*(Fi.T - T) 

        dTcdt = c.betta*(T-Tc) + c.gamma*Q

        return [dcAdt, dcBdt, dcCdt, dcDdt, dTdt, dTcdt]
        
    def evaluate(self):        
        self.set_result("F", self._x.F)
        self.set_result("Tc", self._x.Tc)
        dt = self._model.dt()

        soln = self.solver(lambda t, x, u, c: self.ode(x, u, c)
                        , [0, self._model.dt()]
                        , [self._x.F.cA,self._x.F.cB, self._x.F.cC, self._x.F.cD, self._x.F.T, self._x.Tc]
                        , args = ( (self._u.F(), self._u.Q(), )
                                 , self.config) )
        
        self._x.F.cA = soln.y[0, :][-1]  
        self._x.F.cB = soln.y[1, :][-1]  
        self._x.F.cC = soln.y[2, :][-1]  
        self._x.F.cD = soln.y[3, :][-1]  
        self._x.F.T  = soln.y[4, :][-1] 
        self._x.Tc   = soln.y[5, :][-1]

class KlattEngell(ProcessModel):
    def __init__(self, solver, *
                , dt = 0.005, init_state = None , observer = None):
        
        if observer is None:
            def step_observer(model, state):
                F   = state['SensorF']['F'] #5
                Tc  = state['SensorTc']['Tc'] #1

                return [ F.cA, F.cB, F.cC, F.cD, F.T, Tc]
            observer = step_observer

        super().__init__("KlattEngell", dt = dt, observer = observer)

        inlet = ProcessInputNode("InletFeed", { "F": KlattEngellReactorNode.IFlow(5100, 377.15, 8.256)})         
        coolant = ProcessInputNode("Coolant"  , { "Q": -6239})
        reactor = KlattEngellReactorNode("Reactor", solver=solver) if (init_state is None) else KlattEngellReactorNode("Reactor", solver=solver, cA = init_state["cA"], cB=init_state["cB"], cC = init_state["cC"], cD = init_state["cD"], T = init_state["T"], Tc = init_state["Tc"])

        self.add_node(inlet)
        self.add_node(coolant)
        self.add_node(reactor)

        self.bond_pins(reactor._u.F, inlet._o.F)
        self.bond_pins(reactor._u.Q, coolant._o.Q)
        
        output_names = reactor.outputs()
        for oname in output_names:
            sensor = Sensor(f"Sensor{oname}", oname)
            self.add_node(sensor)
            self.bond_nodes(sensor.name(), oname , reactor.name(), oname)

#==========================================================================================
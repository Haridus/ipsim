from ipsim import *
from numpy import sqrt, power
from copy import deepcopy

#==========================================================
def threashold_value(v, *, min = 0, max = 1):
    return min if v < min else (max if v > max else v)

def threashold_value_min(v,*, min = 0):
    return min if v < min else v

class STEPExFlow:
    A = 0
    B = 1
    C = 2
    D = 3

    def __init__(self, components, T, nominal_flow):
        self.Comp = deepcopy(components) # components
        self.T    = T # temperature Kelvins
        self.Fn   = nominal_flow # nominal flow
        self.F    = self.Fn  # effective flow

class STEPExReactorNode(ProcessNode):
    I  = Inputs(  ("F1", "F2", "X", "D",) )
    O  = Outputs( ("f1", "f2", "F3", "F4", "X", "P","Vl",) )
    U  = States(  ("N", "X", ) )

    class Config:
        def __init__(self):
            self.rhoD   = 8.3     # kmol/m3 density of D
            self.k0     = 0.00117 # Constant for the isothermal operation
            self.thau_v = 1*(10/3600) # h
            self.KcVL   = -1.4
            self.Cv3    = 0.00352*100 #value multiplied by 100 because we use valse positions from 0..1 instead 0..100
            self.Cv4    = 0.0417*100
            self.V = 122 # m3 reactor volume
            self.VlMax = 30 # m3 reactor volume
            self.T = 373 # K reactor temperature
            self.u4bar = 0.4703024823457651 
            self.Rkcal = 1.987 # cal/(mol*K) constant for perfect gases in calories
            self.Rkj   = 8.31451 # J/(mol*K) constant for perfect gases in joulies
            self.T0c    = 273.15  # K 0C in Kelvin

    def __init__(self, name, *, config = Config(), solver = None 
                 , N  = [44.49999958429348, 13.53296996509594, 36.64788062995841, 110.0] #kmol - starting total molar holdup
                 , X = [0.6095327313484253, 0.2502232231706676, 0.392577701760644, 0.4703024823457651] 
                 
                 ):
        super().__init__(name)
        self.config = config
        self.solver = solver
        self._x.N  = deepcopy(N)
        self._x.X  = deepcopy(X)
    
    def ode(self, x, u, c):
        nA, nB, nC, nD, uX1, uX2, uX3, uX4 = x
        F1, F2, u1, u2, u3, u4, D = u
        
        uX1 = threashold_value(uX1)
        uX2 = threashold_value(uX2)
        uX3 = threashold_value(uX3)
        uX4 = threashold_value(uX4)

        Vd = nD/c.rhoD
        Vv = c.V - Vd

        pa = nA*c.Rkj*c.T/Vv
        pb = nB*c.Rkj*c.T/Vv
        pc = nC*c.Rkj*c.T/Vv
        P = pa+pb+pc
   
        Fi1 = F1.Fn*uX1*D[3]
        Fi2 = F2.Fn*uX2*D[4]

        Fi3 = uX3*c.Cv3*sqrt(P-100) if P-100 > 0 else 0
        Fi4 = uX4*c.Cv4*sqrt(P-100) if P-100 > 0 else 0

        # The u(4) value is the setpoint for the liquid level, going to an analog P controller.
        VLpct=Vd/c.VlMax
        _u4=c.u4bar+c.KcVL*(u4-VLpct)

        yA1_ef = threashold_value(F1.Comp[STEPExFlow.A] - D[0] - 0.5*D[1])
        yB1_ef = threashold_value(F1.Comp[STEPExFlow.B] + D[1])
        yC1_ef = threashold_value(F1.Comp[STEPExFlow.C] + D[0] - 0.5*D[1])

        R =    c.k0*power(pa,1.2)*power(pc,0.4)*D[2] if ((pa > 0) and (pc > 0)) else 0
        dNa = (yA1_ef*Fi1 + F2.Comp[STEPExFlow.A]*Fi2 - Fi3*pa/P - R)
        dNb = (yB1_ef*Fi1 - Fi3*pb/P)
        dNc = (yC1_ef*Fi1 - Fi3*pc/P - R)
        dNd = (R - Fi4)

        duX1 = D[5]*(u1 - uX1) /c.thau_v
        duX2 = D[6]*(u2 - uX2) /c.thau_v
        duX3 = D[7]*(u3 - uX3) /c.thau_v
        duX4 = D[8]*(_u4 - uX4)/c.thau_v

        return [dNa,dNb,dNc,dNd,duX1,duX2,duX3,duX4]

    def evaluate(self):
        dt = self._model.dt()
        N = self._x.N
        X = self._x.X
        D = self._u.D()

        nA = N[STEPExFlow.A]
        nB = N[STEPExFlow.B]
        nC = N[STEPExFlow.C]
        nD = N[STEPExFlow.D]

        X1 = threashold_value(X[0])
        X2 = threashold_value(X[1])
        X3 = threashold_value(X[2])
        X4 = threashold_value(X[3])
        
        Vd = nD/self.config.rhoD
        Vv = self.config.V - Vd

        pa = nA*self.config.Rkj*self.config.T/Vv
        pb = nB*self.config.Rkj*self.config.T/Vv
        pc = nC*self.config.Rkj*self.config.T/Vv
        P = pa+pb+pc

        F1 = self._u.F1()
        F2 = self._u.F2()
        Fi1 = F1.Fn*X1*D[3]
        Fi2 = F2.Fn*X2*D[4]
        Fi3 = X3*self.config.Cv3*sqrt(P-100) if P-100 > 0 else 0
        Fi4 = X4*self.config.Cv4*sqrt(P-100) if P-100 > 0 else 0

        F3 = STEPExFlow((pa/P, pb/P, pc/P, 0), self.config.T, Fi3)
        F3.F = Fi3

        F4 = STEPExFlow((0,0,0,1), self.config.T, Fi4)
        F4.F = Fi4

        self.set_result("f1", Fi1)
        self.set_result("f2", Fi2)
        self.set_result("F3", F3)
        self.set_result("F4", F4)
        self.set_result("Vl", Vd/self.config.VlMax)
        self.set_result("P", P) 
        self.set_result("X", self._x.X) 

        #evaluate state change
        uX1 = threashold_value(self._u.X()[0])
        uX2 = threashold_value(self._u.X()[1])
        uX3 = threashold_value(self._u.X()[2])
        uX4 = threashold_value(self._u.X()[3])
        
        soln = self.solver(lambda t, x, u, c: self.ode(x, u, c)
                        , [0, self._model.dt()]
                        , [ self._x.N[STEPExFlow.A], self._x.N[STEPExFlow.B], self._x.N[STEPExFlow.C], self._x.N[STEPExFlow.D]
                          , self._x.X[0], self._x.X[1], self._x.X[2], self._x.X[3] ]
                        , (F1, F2, uX1, uX2, uX3, uX4, D, )
                        , self.config )

        N[STEPExFlow.A] = threashold_value_min(soln.y[0, :][-1])  
        N[STEPExFlow.B] = threashold_value_min(soln.y[1, :][-1])  
        N[STEPExFlow.C] = threashold_value_min(soln.y[2, :][-1])  
        N[STEPExFlow.D] = threashold_value_min(soln.y[3, :][-1])  
        X[0] = threashold_value(soln.y[4, :][-1]) 
        X[1] = threashold_value(soln.y[5, :][-1]) 
        X[2] = threashold_value(soln.y[6, :][-1]) 
        X[3] = threashold_value(soln.y[7, :][-1])    

class STEPEx(ProcessModel):
    def __init__(  self, solver, *
               , dt = 0.1
               , init_state = None 
               , observer = None
               , manipulator = None
               , config = STEPExReactorNode.Config()):
        
        if observer is None:
            def step_observer(model, state):
                f1 = state['Sensorf1']['f1'] #1
                f2 = state['Sensorf2']['f2'] #1
                F3 = state['SensorF3']['F3'] #3
                F4 = state['SensorF4']['F4'] #2
                Vl = state['SensorVl']['Vl'] #1
                P  = state['SensorP']['P']   #1
                X  = state['SensorX']['X']   #4

                return [ X[0], X[1], X[2], X[3]
                       , f1, f2, F3.F, F4.F
                       , F3.Comp[STEPExFlow.A], F3.Comp[STEPExFlow.B], F3.Comp[STEPExFlow.C]
                       , P, Vl ]
            observer = step_observer

        if manipulator is None:
            manipulator = ProcessModel.make_common_manipulator([("ValvesControl","X"), ])

        super().__init__("STEP", dt = dt, observer = observer, manipulator = manipulator)
        
        F1 = STEPExFlow([0.485, 0.005,0.51,0],373, 330.46)
        F2 = STEPExFlow([1,0,0,0],373,22.46)
        uX = [0.6095327313484253, 0.2502232231706676, 0.392577701760644, 0.4703024823457651]
        D  = [0, 0, 1, 1, 1, 1, 1, 1, 1] 

        flow_1 = ProcessInputNode("Stream1", {"F":F1})
        flow_2 = ProcessInputNode("Stream2", {"F":F2})
        valve_control = ProcessInputNode("ValvesControl", {"X":uX})
        disturbances_control = ProcessInputNode("Disturbances", {"D":D})
        reactor   = STEPExReactorNode("STEP", solver=solver, config=config) if (init_state is None) else STEPExReactorNode ("STEP", solver=solver, config=config, N = init_state["N"], X = init_state["X"] )
        
        self.add_node(flow_1)
        self.add_node(flow_2)
        self.add_node(valve_control)
        self.add_node(disturbances_control)
        self.add_node(reactor)

        self.bond_pins(reactor._u.F1 , flow_1._o.F)
        self.bond_pins(reactor._u.F2 , flow_2._o.F)
        self.bond_pins(reactor._u.X  , valve_control._o.X)
        self.bond_pins(reactor._u.D  , disturbances_control._o.D)

        output_names = reactor.outputs()
        for oname in output_names:
            sensor = Sensor(f"Sensor{oname}", oname)
            self.add_node(sensor)
            self.bond_nodes(sensor.name(), oname , reactor.name(), oname)

#===============================================================================
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../ipsim")) 

from ipsim import *

import numpy as np
from copy import deepcopy

#=============================================================================
class DistillationColumnNode(ProcessNode):
    I  = Inputs(  ("rr", "x_feed", "feed",) )
    O  = Outputs( ("xd", "xb",) )
    U  = States(  ("x",) )

    class Config:
        def __init__(self):
            self.vol     = 1.6   # Relative Volatility = (yA/xA)/(yB/xB) = KA/KB = alpha(A,B)
            self.atray   = 0.25  # Total Molar Holdup in the Condenser
            self.acond   = 0.5   # Total Molar Holdup on each Tray
            self.areb    = 1.0   # Total Molar Holdup in the Reboiler

    def __init__(self, name, *, config = Config(), solver = None
                , x = np.array([ 0.935,0.900,0.862,0.821,0.779,0.738, \
                                 0.698,0.661,0.628,0.599,0.574,0.553,0.535,0.521,    \
                                 0.510,0.501,0.494,0.485,0.474,0.459,0.441,0.419,    \
                                 0.392,0.360,0.324,0.284,0.243,0.201,0.161,0.125,    \
                                 0.092,0.064]) ):
        super().__init__(name)
        self._x.x = deepcopy(x)
        self.config = config
        self.solver = solver
        
    def ode(self, x, u, c):
        """
        States (32):
        x(0) - Reflux Drum Liquid Mole Fraction of Component A
        x(1) - Tray 1 - Liquid Mole Fraction of Component A
        .
        .
        .
        x(16) - Tray 16 - Liquid Mole Fraction of Component A (Feed)
        .
        .
        .
        x(30) - Tray 30 - Liquid Mole Fraction of Component A
        x(31) - Reboiler Liquid Mole Fraction of Component A
        """
        rr, Feed, x_Feed = u
        D=0.5*Feed     # Distillate Flowrate (mol/min)
        L=rr*D         # Flowrate of the Liquid in the Rectification Section (mol/min)
        V=L+D          # Vapor Flowrate in the Column (mol/min)
        FL=Feed+L      # Flowrate of the Liquid in the Stripping Section (mol/min)
        y = np.empty(len(x))
        for i in range(32):
            y[i] = x[i] * c.vol/(1.0+(c.vol-1.0)*x[i])
            
        # Compute xdot
        xdot = np.empty(len(x))
        xdot[0] = 1/c.acond*V*(y[1]-x[0])
        for i in range(1,16):
            xdot[i] = 1.0/c.atray*(L*(x[i-1]-x[i])-V*(y[i]-y[i+1]))
        xdot[16] = 1/c.atray*(Feed*x_Feed+L*x[15]-FL*x[16]-V*(y[16]-y[17]))
        for i in range(17,31):
            xdot[i] = 1.0/c.atray*(FL*(x[i-1]-x[i])-V*(y[i]-y[i+1]))
        xdot[31] = 1/c.areb*(FL*x[30]-(Feed-D)*x[31]-V*y[31])
        return xdot
    
    def evaluate(self):   
        soln = self.solver(lambda t, x, u, c: self.ode(x, u, c)
                        , [0, self._model.dt()]
                        , self._x.x
                        , (self._u.rr(), self._u.feed(), self._u.x_feed(), )
                        , self.config )
        
        res = soln.y[:,-1]
        self._x.x = res

        self.set_result("xd",self._x.x[0])
        self.set_result("xb",self._x.x[31])

class DistillationColumnFeed(ProcessNode):
    O  = Outputs( ("feed", "x",) )
    U  = States(  ("feed", "x",) )

    def __init__(self, name):
        import scipy
        import itertools
        super().__init__(name)
        
        num_x_feed, den_x_feed = scipy.signal.butter(2, 0.05)
        self.dis_x = itertools.cycle(scipy.signal.lfilter(num_x_feed, den_x_feed, np.random.normal(0, 0.05, 5000)))
        self.disruptor = lambda: next(self.dis_x)
        self._x.feed = 1
        self._x.x    = 0.42
                
    def _evaluate(self):
        if (self._model is None) or (self._model.time() != self._current_time):
            self._results.clear()

        if not self._results:
            self.set_result("x", self._x.x+self.disruptor())
            self.set_result("feed",self._x.feed)

class DistillationColumn(ProcessModel):
    def __init__( self, solver, *
                , dt = 1
                , init_state = None
                , observer = None
                , manipulator = None):
        super().__init__( "DistillationColumn"
                        , dt = dt
                        , observer = observer, manipulator = manipulator )

        feed   = DistillationColumnFeed("Feed")
        reflux = ProcessInputNode("Reflux", {"ratio":3})
        column = DistillationColumnNode("DistillationColumn", solver=solver) if (init_state is None) else DistillationColumnNode("DistillationColumn",x=init_state["x"], solver=solver)

        sensorXd = Sensor("SensorXd","xd")
        sensorXb = Sensor("SensorXb","xb")
        
        self.add_node(feed)
        self.add_node(reflux)
        self.add_node(column)
        self.add_node(sensorXd)
        self.add_node(sensorXb)

        self.bond_nodes("DistillationColumn","feed","Feed","feed")
        self.bond_nodes("DistillationColumn","x_feed","Feed","x")
        self.bond_nodes("DistillationColumn","rr","Reflux","ratio")

        self.bond_nodes("SensorXd","xd", "DistillationColumn", "xd")
        self.bond_nodes("SensorXb","xb", "DistillationColumn", "xb")

#========================================================================


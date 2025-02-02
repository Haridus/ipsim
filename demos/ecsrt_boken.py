''' Present an interactive function explorer with slider widgets.

Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve sliders.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/sliders

in your browser.

'''

#----------------------------------------------------------------
import os
import sys
import pathlib
#sys.path.append(os.path.join(os.path.dirname(__file__), "../../../ipsim/ipsim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../ipsim/"))  
print(sys.path)

from ipsim import *

#----------------------------------------------------------------
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from scipy.integrate import solve_ivp

from copy import deepcopy

#----------------------------------------------------------------
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Button
from bokeh.plotting import figure
from bokeh.client import push_session
from bokeh.embed import server_session
#----------------------------------------------------------------
class ECSRTTest:
    Ea  = 72750     # activation energy J/gmol
    R   = 8.314     # gas constant J/gmol/K
    k0  = 7.2e10    # Arrhenius rate constant 1/min
    dHr = -5.0e4    # Enthalpy of reaction [J/mol]
    rho = 1000.0    # Density [g/L]
    Cp  = 0.239     # Heat capacity [J/g/K]
    UA  = 5.0e4     # Heat transfer [J/min/K]
    
    V   = 100.0     # Volume [L]
    q   = 100.0     # Flowrate [L/min]
    cAi = 1.0       # Inlet feed concentration [mol/L]
    Ti  = 350.0     # Inlet feed temperature [K]
    Tc  = 305.0     # Coolant temperature [K]

    cA0 = 0.5;      # Initial concentration [mol/L]
    T0  = 350.0;    # Initial temperature [K]
    
    def k(T):
        return ECSRTTest.k0*np.exp(-ECSRTTest.Ea/ECSRTTest.R/T)

class ExothermicContinuousStirredTankReactor0(ProcessNode):
    def __init__(self, name, *, cA0 = ECSRTTest.cA0, T0 = ECSRTTest.T0, V = ECSRTTest.V
                , dHr = ECSRTTest.dHr, rho = ECSRTTest.rho, Cp=ECSRTTest.Cp,UA = ECSRTTest.UA):
        super().__init__(name)
        self._cA = deepcopy(cA0)
        self._T = deepcopy(T0)
        self._cB = 0
        self._V = V
        self._dHr = dHr
        self._rho=rho
        self._Cp = Cp
        self._UA = UA
        self.create_input("q")
        self.create_input("cAi")
        self.create_input("Ti")
        self.create_input("Tc")
        self.create_output("cA")
        self.create_output("T")
        self.create_output("cB")
        self._calltimes = 0
    
    def __call__(self, t, y):
        i = self.inputs()
        q   = i["q"]()
        cAf = i["cAi"]()
        Tf  = i["Ti"]()
        Tc  = i["Tc"]()

        cA, T, cB = y
       
        qV    = q/self._V
        kCa = ECSRTTest.k(T)*cA
        dHpC  = -self._dHr/(self._rho*self._Cp)
        UAVpC = self._UA/(self._V*self._rho*self._Cp)

        dcAdt = qV*(cAf - cA) - kCa
        dTdt  = qV*(Tf - T) + dHpC*kCa + UAVpC*(Tc-T)
        dBdt  = qV*(0 - cB) + kCa

        return [dcAdt, dTdt, dBdt]
        
    def evaluate(self):
        self._calltimes = self._calltimes+1
        
        i = self.inputs()
        q   = i["q"]()
        cAf = i["cAi"]()
        Tf  = i["Ti"]()
        Tc  = i["Tc"]()
        cA  = self._cA
        T   = self._T
        cB  = self._cB
        
        dt = self._model.dt()

        t_eval = np.linspace(0, dt, 50)
        soln = solve_ivp(self, [min(t_eval), max(t_eval)], [cA, T, cB], t_eval=t_eval)
        
        cA = soln.y[0, :][-1]
        T  = soln.y[1, :][-1]
        cB = soln.y[2, :][-1]
        
        self._cA = cA
        self._T = T
        self._cB = cB
        
        self.set_result("cA",cA)
        self.set_result("cB",cB)
        self.set_result("T",T)
        
def prepare_model(dt = 10.0/2000):    
    process_model = ProcessModel("test",dt=dt)
    process_model.add_node(ProcessInputNode("InletFeed"
                                         , {"Flowrate":100,"Concentration":1,"Temperature":350}))
    process_model.add_node(ProcessInputNode("Coolant"
                                          , {"Temperature":305}))
    process_model.add_node(ExothermicContinuousStirredTankReactor0("ECSTR"))
    process_model.bond_nodes("ECSTR","q","InletFeed","Flowrate")
    process_model.bond_nodes("ECSTR","cAi","InletFeed","Concentration")
    process_model.bond_nodes("ECSTR","Ti","InletFeed","Temperature")
    process_model.bond_nodes("ECSTR","Tc","Coolant","Temperature")
    
    return process_model

#----------------------------------------------------------------
dt = 0.01
ECSTR = prepare_model(dt)
time_current = 0
time_max = 60 

#----------------------------------------------------------------
# Set up data
source_A = ColumnDataSource(data=dict(x=[], y=[]))
source_B = ColumnDataSource(data=dict(x=[], y=[]))
source_T = ColumnDataSource(data=dict(x=[], y=[]))

# Set up plot
plot_A = figure(height=200, width=800, title="Concentration of A",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, time_max], y_range=[0, 1])
plot_A.line('x', 'y', source=source_A, line_width=3, line_alpha=0.6)

plot_B = figure(height=200, width=800, title="Concentration of B",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, time_max], y_range=[0, 1])
plot_B.line('x', 'y', source=source_B, line_width=3, line_alpha=0.6)

plot_T = figure(height=200, width=800, title="Temperature",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, time_max], y_range=[200, 500])
plot_T.line('x', 'y', source=source_T, line_width=3, line_alpha=0.6)

reset_button = Button(label="Reset",button_type="primary")
coolant_temperature = Slider(title="CoolantTemperature", value=305.0, start=300.0, end=310.0, step=0.5)

def reset():
    global time_current
    ECSTR = prepare_model(dt)
    time_current = 0
    source_A.data=dict(x=[], y=[])
    source_B.data=dict(x=[], y=[])
    source_T.data=dict(x=[], y=[])

    a = coolant_temperature.value
    ECSTR.nodes()["Coolant"].change_value("Temperature",a)

def update_data(attrname, old, new):
    a = coolant_temperature.value
    ECSTR.nodes()["Coolant"].change_value("Temperature",a)

def update():
    global time_current
    if time_current > time_max:
        reset()

    points = int(1/dt)
    x = np.linspace(time_current,time_current+1,points)
    time_current = time_current + 1
    
    cAs = np.zeros(points)
    cBs = np.zeros(points)
    Ts = np.zeros(points)
    for _ in range(points):
        state = ECSTR.next_state(("ECSTR",))
        cAs[_] = state['ECSTR']['cA']
        cBs[_] = state['ECSTR']['cB']
        Ts[_] = state['ECSTR']['T']

    new_data = dict(x=x, y=cAs)
    source_A.stream(new_data)
    new_data = dict(x=x, y=cBs)
    source_B.stream(new_data)
    new_data = dict(x=x, y=Ts)
    source_T.stream(new_data)
    
for w in [coolant_temperature]:
    w.on_change('value', update_data)
reset_button.on_click(reset)

inputs = column(coolant_temperature, reset_button)
plots = column(plot_A,plot_B,plot_T)
curdoc().add_root(row(inputs, plots, width=1024))
curdoc().add_periodic_callback(update, 1000)
curdoc().title = "ECSTR"

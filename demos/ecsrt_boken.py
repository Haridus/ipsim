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
import random as rnd

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
#----------------------------------------------------------------
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Button, CheckboxGroup, Toggle
from bokeh.plotting import figure
from bokeh.client import push_session
from bokeh.embed import server_session
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Div

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

        t_eval = np.linspace(0, dt, 10)
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

class NoisySensor(Sensor):
    def __init__(self, name, input_name, *, dev_prob = 0, dev_amp = 0):
        super().__init__(name, input_name)
        self._dev_prob = dev_prob
        self._dev_amp = dev_amp
        self._input_name = input_name

    def evaluate(self):
        value = self.inputs()[self._input_name]()
        if (self._dev_prob > 0) and (self._dev_amp > 0):
            prob = rnd.uniform(0,1)
            if prob < self._dev_prob:
                amp = rnd.uniform(-self._dev_amp, self._dev_amp)
                value = value + amp
                value = value if value > 0 else 0 
                
        self.set_result(self._input_name, value)

    def change_value(self, name, value):
        if name == "dev_prob":
            self._dev_prob = value
        if name == "dev_amp":
            self._dev_amp = value

def prepare_model(dt = 10.0/2000):    
    process_model = ProcessModel("test",dt=dt)
    process_model.add_node(ProcessInputNode("InletFeed"
                                         , {"Flowrate":100,"Concentration":1,"Temperature":350}))
    process_model.add_node(ProcessInputNode("Coolant"
                                          , {"Temperature":290}))
    process_model.add_node(ExothermicContinuousStirredTankReactor0("ECSTR"))
    process_model.bond_nodes("ECSTR","q","InletFeed","Flowrate")
    process_model.bond_nodes("ECSTR","cAi","InletFeed","Concentration")
    process_model.bond_nodes("ECSTR","Ti","InletFeed","Temperature")
    process_model.bond_nodes("ECSTR","Tc","Coolant","Temperature")
    
    process_model.add_node(NoisySensor("SensorA","cA"))
    process_model.add_node(NoisySensor("SensorB","cB"))
    process_model.add_node(NoisySensor("SensorT","T"))
    
    process_model.bond_nodes("SensorA","cA", "ECSTR", "cA")
    process_model.bond_nodes("SensorB","cB", "ECSTR", "cB")
    process_model.bond_nodes("SensorT","T", "ECSTR", "T")

    sensorA =  process_model.nodes()["SensorA"]
    sensorB =  process_model.nodes()["SensorB"]
    sensorT =  process_model.nodes()["SensorT"]

    sensorA.change_value("dev_prob", 1)
    sensorA.change_value("dev_amp", 0.05)

    sensorB.change_value("dev_prob", 1)
    sensorB.change_value("dev_amp", 0.05)
    
    sensorT.change_value("dev_prob", 1)
    sensorT.change_value("dev_amp", 10)

    return process_model

#----------------------------------------------------------------
def run_model_steps_ex(process_model, Tcs, iterations_per_step): 
    observed_nodes = ("SensorA","SensorB","SensorT",)

    parameters_metadata = (
          {"parameter":"cA"
           , "range":(0,1)
           , "units": ""
           , "title":"Concentration of A"
           , "sensor_node": "SensorA"}
        , {"parameter":"cB"
           , "range":(0,1)
           , "units": ""
           , "title":"Concentration of B"
           , "sensor_node": "SensorB"}
        , {  "parameter":"T"
           , "range":(300,600)
           , "units": "K"
           , "title":"Temperature"
           , "sensor_node": "SensorT" }
    )

    iterations = int(len(Tcs)*iterations_per_step)
    x = np.empty((iterations,len(parameters_metadata)))
    y = np.empty((iterations,1),dtype=np.dtype('B'))

    Tci = -1
    Tc  = 0
    for _ in range(iterations):
        if _ % iterations_per_step == 0:
            Tci = 0 if Tci == -1 else Tci + 1
            Tc = Tcs[Tci]                 
            process_model.nodes()["Coolant"].change_value("Temperature", Tc)
        state = process_model.next_state(observed_nodes)
        data =  []
        for metadata in parameters_metadata:
             data.append(state[metadata['sensor_node']][metadata["parameter"]])

        x[_] =  data
        y[_] = 0 if (Tc < 304 or Tc > 307) else 1

    return x, parameters_metadata, y

ML_MODEL_LOGISTIC = "Logistic Regression"
ML_MODEL_RF = "Random Forest"
ML_MODEL_GB = "Gradient Boosting"
ML_MODEL_ADAB = "AdaBoost"

def train_logistic(x,y):
    model = LogisticRegression()
    model.fit(x, y.ravel())
    return model

def train_random_forest(x,y):
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(x, y.ravel())
    return model

def train_gradient_boostins(x,y):
    model = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=1, random_state=0)
    model.fit(x, y.ravel())
    return model

def train_ada_boostins(x,y):
    model = AdaBoostClassifier(n_estimators=500, algorithm="SAMME", random_state=0)
    model.fit(x, y.ravel())
    return model

def train_bagging(x,y):
    model = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0)
    model.fit(x, y.ravel())
    return model

def show_model_statistics(y, y_pred, model_name):        
    f_metrics = {"Accuracy":accuracy_score
                ,"Precision":precision_score
                ,"Recall":recall_score
                ,"F1" : f1_score
                ,"Cohens kappa": cohen_kappa_score
                ,"ROC AUC": roc_auc_score
                ,"Confusion matrix": confusion_matrix}
    print(f"{model_name}:")
    for metric in f_metrics:
        try:
            print(metric, f_metrics[metric](y, y_pred))
        except Exception as e:
            print(f"Metiric {metric}: {e}")

def train_ml_models():
    dt = 0.1
    process_model = prepare_model(dt)
    
    sensorA =  process_model.nodes()["SensorA"]
    sensorB =  process_model.nodes()["SensorB"]
    sensorT =  process_model.nodes()["SensorT"]

    sensorA.change_value("dev_prob", 1)
    sensorA.change_value("dev_amp", 0.05)

    sensorB.change_value("dev_prob", 1)
    sensorB.change_value("dev_amp", 0.05)
    
    sensorT.change_value("dev_prob", 1)
    sensorT.change_value("dev_amp", 10) 

    observed_nodes = ("SensorA","SensorB","SensorT",)
    for _ in range(200): # skip first iteration to reach steady-state 
        process_model.next_state(observed_nodes)

    x, metadata, y = run_model_steps_ex(process_model,[300, 303, 304, 305, 306, 308, 310], 200)
    #show_data(x, metadata, dt)
    count_true = 0
    for yi in y:
        if yi == 0:
            count_true += 1
    print(f"normal class: {count_true} faults: {len(y)-count_true} total: {len(y)}") 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    models = {ML_MODEL_LOGISTIC:train_logistic
             ,ML_MODEL_RF:train_random_forest
             ,ML_MODEL_GB:train_gradient_boostins
             ,ML_MODEL_ADAB: train_ada_boostins
             }

    trained_models = {}
    for title, func in models.items():
        ml_model = func(x_train,y_train)
        y_pred   = ml_model.predict(x_test)
        show_model_statistics(y_test,y_pred, "")
        trained_models[title] = ml_model

    return trained_models

#----------------------------------------------------------------
dt = 0.1
ECSTR = prepare_model(dt)
ML_MODELS = train_ml_models()
time_current = 0
time_max = 60 
USE_ML_MODEL = ""

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
coolant_temperature = Slider(title="Coolant Temperature", value=304.0, start=295.0, end=350.0, step=0.5)
ml_detected_faults_label = Div(text=f"""<p>Faults Detected:</p>""", width=200)
ml_detected_faults = Div(text=f"""<p>{ML_MODEL_LOGISTIC:} {0}</p>
                         <p>{ML_MODEL_RF:} {0}</p>
                         <p>{ML_MODEL_GB:} {0}</p>
                         <p>{ML_MODEL_ADAB:} {0}</p>
                         """, width=200)

ml_switch_label_faults = Div(text=f"""<p>Use Model:</p>""")
no_model_toggle = Toggle(label="No Model", active=True)
logistic_regression_toggle = Toggle(label=ML_MODEL_LOGISTIC, active=False)
random_forest_toggle = Toggle(label=ML_MODEL_RF, active=False )
gradient_boosting_toggle = Toggle(label=ML_MODEL_GB, active=False)
ada_boosting_toggle = Toggle(label=ML_MODEL_ADAB, active=False)

def reset():
    global time_current
    global ECSTR
    ECSTR = prepare_model(dt)
    time_current = 0
    source_A.data=dict(x=[], y=[])
    source_B.data=dict(x=[], y=[])
    source_T.data=dict(x=[], y=[])

    a = coolant_temperature.value
    print(f"reset temperature {a}")
    ECSTR.nodes()["Coolant"].change_value("Temperature",a)

    for _ in range(200): # skip first iteration to reach steady-state 
        ECSTR.next_state(('ECSTR',))

def update_data(attrname, old, new):
    a = coolant_temperature.value
    ECSTR.nodes()["Coolant"].change_value("Temperature",a)

def use_ml_model(attrname, old, new):
    global USE_ML_MODEL
    USE_ML_MODEL = False if len(new) == 0 else True
    print(f"use ml model: {USE_ML_MODEL}")
    
def update():
    global ML_MODELS
    global time_current
    ml_models_list = ( ML_MODEL_LOGISTIC
                     , ML_MODEL_RF
                     , ML_MODEL_GB
                     , ML_MODEL_ADAB, )
    
    if time_current > time_max:
        reset()

    points = int(1/dt)
    x = np.linspace(time_current,time_current+1,points)
    time_current = time_current + 1

    faults_detected = {
                       ML_MODEL_LOGISTIC : 0
                     , ML_MODEL_RF : 0
                     , ML_MODEL_GB : 0
                     , ML_MODEL_ADAB : 0,
    }

    cAs = np.zeros(points)
    cBs = np.zeros(points)
    Ts = np.zeros(points)
    faults_x = np.empty((1,3))
    for _ in range(points):
        state = ECSTR.next_state(('SensorA', 'SensorB', 'SensorT'))
        cAs[_] = state['SensorA']['cA']
        cBs[_] = state['SensorB']['cB']
        Ts[_] = state['SensorT']['T']
        faults_x[0] = (state['SensorA']['cA'],state['SensorB']['cB'],state['SensorT']['T'],)

        for model_name in ml_models_list:
            ml_model = ML_MODELS[model_name]
            faults_pred = ml_model.predict(faults_x)
            faults_detected[model_name] += faults_pred[0]

            if (USE_ML_MODEL == model_name) and (faults_detected[model_name]==1):
                coolant_temperature.value = coolant_temperature.value+1
                ECSTR.nodes()["Coolant"].change_value("Temperature", coolant_temperature.value)

    faults_detected_text = ""
    for model_name in ml_models_list:
        faults_detected_text += f"""<p>{model_name}: {faults_detected[model_name]}</p>"""
        if (USE_ML_MODEL == model_name) and (faults_detected[model_name] == 0):
            coolant_temperature.value = coolant_temperature.value-1
            ECSTR.nodes()["Coolant"].change_value("Temperature", coolant_temperature.value)
    ml_detected_faults.text = faults_detected_text

    new_data = dict(x=x, y=cAs)
    source_A.stream(new_data)
    new_data = dict(x=x, y=cBs)
    source_B.stream(new_data)
    new_data = dict(x=x, y=Ts)
    source_T.stream(new_data)
    
for w in [coolant_temperature]:
    w.on_change('value', update_data)

useMLModel = CheckboxGroup(labels=["Use ML Model", ], active=[])
useMLModel.on_change('active', use_ml_model)

ECSTR_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Agitated_vessel.svg/500px-Agitated_vessel.svg.png"
ldiv_image = Div(text=f"""<img src="{ECSTR_logo}" alt="div_image" width="300" height="400" >""")
image_label = Div(text=f"""<p style="text-align: center"><a href="{ECSTR_logo}">Image source</a></p>""")

#----------------------------------------------------------------
reset()

def not_use_ml_model(attrname, old, new):
    global USE_ML_MODEL
    if new == True:
        #no_model_toggle.active = False
        logistic_regression_toggle.active = False
        random_forest_toggle.active = False
        gradient_boosting_toggle.active = False
        ada_boosting_toggle.active = False
        USE_ML_MODEL = ""
    else:
        not_use_ml_model.active = True

def use_logistic_regression_model(attrname, old, new):
    global USE_ML_MODEL
    if new == True:
        no_model_toggle.active = False
        #logistic_regression_toggle.active = False
        random_forest_toggle.active = False
        gradient_boosting_toggle.active = False
        ada_boosting_toggle.active = False
        USE_ML_MODEL = ML_MODEL_LOGISTIC

def use_random_forest_model(attrname, old, new):
    global USE_ML_MODEL
    if new == True:
        no_model_toggle.active = False
        logistic_regression_toggle.active = False
        #random_forest_toggle.active = False
        gradient_boosting_toggle.active = False
        ada_boosting_toggle.active = False
        USE_ML_MODEL = ML_MODEL_RF

def use_gradiest_boosting_model(attrname, old, new):
    global USE_ML_MODEL
    if new == True:
        no_model_toggle.active = False
        logistic_regression_toggle.active = False
        random_forest_toggle.active = False
        #gradient_boosting_toggle.active = False
        ada_boosting_toggle.active = False
        USE_ML_MODEL = ML_MODEL_GB

def use_ada_boosting_model(attrname, old, new):
    global USE_ML_MODEL
    if new == True:
        no_model_toggle.active = False
        logistic_regression_toggle.active = False
        random_forest_toggle.active = False
        gradient_boosting_toggle.active = False
        #ada_boosting_toggle.active = False
        USE_ML_MODEL = ML_MODEL_ADAB

no_model_toggle.on_change('active', not_use_ml_model)
logistic_regression_toggle.on_change('active', use_logistic_regression_model)
random_forest_toggle.on_change('active', use_random_forest_model)
gradient_boosting_toggle.on_change('active', use_gradiest_boosting_model)
ada_boosting_toggle.on_change('active', use_ada_boosting_model)


inputs = column(coolant_temperature
               , ml_detected_faults_label 
               , ml_detected_faults
               , ml_switch_label_faults
               , row( no_model_toggle
               , logistic_regression_toggle
               , random_forest_toggle)
               , row(gradient_boosting_toggle
               , ada_boosting_toggle) 
               , ldiv_image
               , image_label)

plots = column(plot_A
               ,plot_B
               ,plot_T)

curdoc().add_root(row(inputs, plots, width=1024))
curdoc().add_periodic_callback(update, 1000)
curdoc().title = "Exotermic Continuous Stirred-Tank Reactor Model"

#----------------------------------------------------------------

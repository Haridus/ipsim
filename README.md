# ipsim: Industrial processes simulator with feedback loop for digital twin benchmarking

## Introduction
Python-based frameworks have experienced significant growth in popularity in recent years, fueled by a mature ecosystem and widespread support from an active open source community. The widespread adoption of Python for data analysis, especially in machine learning-based projects, has generated significant motivation to adapt existing research software and modeling tools for seamless integration into the Python environment. Today, many tools can be developed in the Python ecosystem, including modeling platforms that enable real-time data generation and direct analysis within feedback loops. One area where this can be applied is modeling discrete manufacturing processes and creating their digital doubles. 
This repository contains an implementation of a simple accessible and easily customizable tool for modeling dynamic manufacturing processes, focusing on the flexibility of modularity and ease of development with the ability to further integrate with AI components and feedback loops in Python.

## Installing

To install `ipsim`, run the following command:
```
pip install git+https://github.com/Haridus/ipsim.git
```

## Usage

```python

from ipsim import ProcessModel, ProcessInputNode

class DublingSensor(Sensor):
        def __init__(self, name, input_name):
            super().__init__(name, input_name)

        def evaluate(self):
            i = self.inputs()
            for key in i:
                self.set_result(key, i[key]()*2)

def prepare_model(dt = 0.01):
    model = ProcessModel("test", dt=0.01)
    model.add_node(ProcessInputNode("Reservoir_Coolant",{"Temperature":300}))
    model.add_node(Sensor("R1CTS1","Temperature"))
    model.add_node(DublingSensor("R1CTS2","Temperature"))
    model.bond_nodes("R1CTS1","Temperature", "Reservoir_Coolant", "Temperature")
    model.bond_nodes("R1CTS2","Temperature", "Reservoir_Coolant", "Temperature")

    return model

loops_count = 5
model = prepare_model()
for i in range(loops_count):
    state = model.next_state(("R1CTS1","R1CTS2"))
    assert state["R1CTS1"]["Temperature"] == 300
    assert state["R1CTS2"]["Temperature"] == 600

```

## Testing

To test the library, run the command `pytest ipsim/tests` from the root directory.

## Running experiments

To reproduce the results from the papers go to benchmarks folder.

[benchmarks/tes_ml.ipynb] - implementation of simplified Tennessee Eastman problem (STEP) by Ricker (10.1016/0959-1524(93)80006-W) and experiment for paper @paper

[benchmarks/ecstr_detect_oscillation.ipynb] - implementation of Continuous Stirred-Tank Reactor(CSTR) and experiment for paper @paper

## Demos
[Github/ecsrt_boken.py] - demo of control of CSTR model by ML model for paper @paper

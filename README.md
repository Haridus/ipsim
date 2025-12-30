# ipsim: Industrial processes simulator with feedback loop for digital twin benchmarking

## Introduction
Python-based frameworks have experienced significant growth in popularity in recent years, fueled by a mature ecosystem and widespread support from an active open source community. The widespread adoption of Python for data analysis, especially in machine learning-based projects, has generated significant motivation to adapt existing research software and modeling tools for seamless integration into the Python environment. Today, many tools can be developed in the Python ecosystem, including modeling platforms that enable real-time data generation and direct analysis within feedback loops. One area where this can be applied is modeling discrete manufacturing processes and creating their digital doubles. 
This repository contains an implementation of a simple accessible and easily customizable tool for modeling dynamic manufacturing processes, focusing on the flexibility of modularity and ease of development with the ability to further integrate with AI components and feedback loops in Python.

## The General idea
The general idea of ​​the framework can be illustrated in the figure. A node-based simulator, using sensor nodes, provides results to an AI algorithm, which can control the process by influencing manipulated variables via input nodes.

![Concept](./images/Concept.png)

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
            super().__init__(name,input_name)

        def evaluate(self):
            i = self.inputs()
            for key in i:
                self.set_result(key, i[key]()*2)

def prepare_model(dt = 0.01):
    model = ProcessModel("test", dt=0.01)
    model.add_node(ProcessInputNode("Reservoir_Coolant",{"Temperature":300}))
    model.add_node(Sensor("R1CTS1","Temperature"))
    model.add_node(DoublingSensor("R1CTS2","Temperature"))
    model.bond_nodes("R1CTS1","Temperature", "Reservoir_Coolant", "Temperature")
    model.bond_nodes("R1CTS2","Temperature", "Reservoir_Coolant", "Temperature")

    return model

loops_count = 5
model = prepare_model()
for i in range(loops_count):
    state = model.next_state(nodes_names=("R1CTS1","R1CTS2"))
    assert state["R1CTS1"]["Temperature"] == 300
    assert state["R1CTS2"]["Temperature"] == 600

```

For more advanced examples look __examples__ directory.

## Testing

To test the library, run the command `pytest ipsim/tests` from the root directory.

## Models
The repository contains several ready-made simulators, they are located in directory __ipsim/models__. For examples of models usage look __examples__ directory. 

## Examples
The .ipynb notebooks are located in the __examples__ directory. 

You can also access the [repository](https://github.com/Haridus/ipsim_experiments) where the developed simulators are integrated into the Open Gym environment for subsequent reinforcement learning.

